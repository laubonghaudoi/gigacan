from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from funasr import AutoModel
from torch.nn.utils.rnn import pad_sequence

from .config import TranscribeConfig


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_segment_batch_size(device: str, batch_size_arg: int) -> int:
    if batch_size_arg > 0:
        return batch_size_arg
    if device.startswith("cuda"):
        return 128
    return 4


def ensure_qwen_source(src_dir: Path, repo_url: str) -> Path:
    if (src_dir / "qwen_asr").is_dir():
        return src_dir
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(src_dir)],
        check=True,
    )
    return src_dir


def ensure_qwen_transformers_compat() -> None:
    try:
        import transformers.utils.generic as generic
        from transformers import modeling_rope_utils
    except Exception:
        return

    if not hasattr(generic, "check_model_inputs"):

        def check_model_inputs(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def qwen_default_rope_parameters(
        config: Any = None,
        device: Any = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        if config is None:
            raise ValueError("config is required for default rope parameters")

        base = float(getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = 1.0

        if layer_type is not None and hasattr(config, "rope_parameters"):
            rope_parameters = getattr(config, "rope_parameters")
            if isinstance(rope_parameters, dict):
                layer_params = rope_parameters.get(layer_type, {})
                if isinstance(layer_params, dict):
                    base = float(layer_params.get("rope_theta", base))
                    partial_rotary_factor = float(
                        layer_params.get("partial_rotary_factor", partial_rotary_factor)
                    )

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = int(config.hidden_size) // int(config.num_attention_heads)

        dim = int(head_dim * partial_rotary_factor)
        if dim < 2:
            dim = 2
        if dim % 2 == 1:
            dim -= 1
            if dim < 2:
                dim = 2

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / float(dim)
            )
        )
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = qwen_default_rope_parameters


def prepare_qwen_runtime(src_dir: Path, repo_url: str) -> None:
    qwen_src_dir = ensure_qwen_source(src_dir, repo_url)
    if str(qwen_src_dir) not in sys.path:
        sys.path.insert(0, str(qwen_src_dir))
    ensure_qwen_transformers_compat()


def resolve_vllm_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda:"):
        _, _, raw_idx = device.partition(":")
        if raw_idx.isdigit():
            idx = int(raw_idx)
            if idx == 0:
                return "cuda:0"
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible and visible != raw_idx:
                raise RuntimeError(
                    "vLLM backend requires CUDA_VISIBLE_DEVICES to match the selected "
                    f"device index. Got --device={device}, "
                    f"CUDA_VISIBLE_DEVICES={visible!r}."
                )
            if not visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = raw_idx
            return "cuda:0"
    raise RuntimeError(
        f"vLLM backend requires a CUDA device, got {device!r}. "
        "Use --asr-engine sensevoice for CPU runs."
    )


def _apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    """Low Frame Rate: stack *lfr_m* frames, subsample by *lfr_n*."""
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = int(
            (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n)
            / 2
            * (T_lfr - last_idx)
        )
        inputs = torch.vstack([inputs] + [inputs[-1:]] * num_padding)
    return inputs.as_strided(sizes, strides).clone().to(torch.float32)


def _apply_cmvn(inputs: torch.Tensor, cmvn: torch.Tensor) -> torch.Tensor:
    """Cepstral Mean and Variance Normalization."""
    dim = inputs.shape[-1]
    means = cmvn[0:1, :dim]
    vars_data = cmvn[1:2, :dim]
    inputs += means.to(inputs.device)
    inputs *= vars_data.to(inputs.device)
    return inputs.to(torch.float32)


@dataclass(slots=True)
class ASRResult:
    text: str


class SenseVoiceASRModel:
    """Thin wrapper to match the existing pipeline `.transcribe(...)` contract."""

    def __init__(
        self,
        model: AutoModel,
        *,
        default_language: str,
        use_itn: bool,
        max_inference_batch_size: int,
        device: str = "cpu",
        fbank_workers: int = 0,
    ) -> None:
        self.model = model
        self.default_language = default_language
        self.use_itn = use_itn
        self.max_inference_batch_size = max(1, int(max_inference_batch_size))
        self.device = device

        # Extract internal components for direct inference, bypassing AutoModel.
        self._sensevoice = getattr(model, "model", None)
        self._frontend = getattr(model, "kwargs", {}).get("frontend")
        self._tokenizer = getattr(model, "kwargs", {}).get("tokenizer")

        self._direct_ready = False
        if self._frontend is not None and self._tokenizer is not None:
            self._fbank_params = {
                "num_mel_bins": self._frontend.n_mels,
                "frame_length": self._frontend.frame_length,
                "frame_shift": self._frontend.frame_shift,
                "dither": self._frontend.dither,
                "energy_floor": 0.0,
                "window_type": self._frontend.window,
                "sample_frequency": self._frontend.fs,
                "snip_edges": self._frontend.snip_edges,
            }
            self._lfr_m: int = self._frontend.lfr_m
            self._lfr_n: int = self._frontend.lfr_n
            self._upscale: bool = getattr(self._frontend, "upsacle_samples", True)
            self._cmvn: torch.Tensor | None = self._frontend.cmvn
            self._blank_id: int = self._sensevoice.blank_id
            self._lid_dict: dict[str, int] = self._sensevoice.lid_dict
            self._textnorm_dict: dict[str, int] = self._sensevoice.textnorm_dict
            self._direct_ready = True

        resolved_workers = (
            fbank_workers if fbank_workers > 0
            else min(12, max(4, os.cpu_count() or 4))
        )
        self._fbank_pool = ThreadPoolExecutor(max_workers=resolved_workers)

    def _extract_single_fbank(self, waveform: np.ndarray) -> torch.Tensor:
        """Compute fbank + LFR + CMVN for one waveform segment.

        This calls torchaudio C++ code that releases the GIL, so it can run in
        parallel via ThreadPoolExecutor.
        """
        tensor = torch.from_numpy(waveform).to(torch.float32)
        if self._upscale:
            tensor = tensor * (1 << 15)
        tensor = tensor.unsqueeze(0)

        params = dict(self._fbank_params)
        waveform_ms = tensor.shape[1] / params["sample_frequency"] * 1000
        params["frame_length"] = min(params["frame_length"], waveform_ms)

        mat = kaldi.fbank(tensor, **params)

        if self._lfr_m != 1 or self._lfr_n != 1:
            mat = _apply_lfr(mat, self._lfr_m, self._lfr_n)
        if self._cmvn is not None:
            mat = _apply_cmvn(mat, self._cmvn)

        return mat

    def extract_features(
        self, waveforms: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute fbank features for a batch of waveforms.

        Uses a thread pool for parallelism since kaldi.fbank releases the GIL.
        Returns (feats_pad, feats_lens) ready for ``transcribe_preprocessed``.
        """
        if not waveforms:
            return torch.zeros(0, 0, 0), torch.zeros(0, dtype=torch.int32)

        futures = [
            self._fbank_pool.submit(self._extract_single_fbank, wf)
            for wf in waveforms
        ]
        feats_list = [f.result() for f in futures]
        feats_lens = torch.tensor(
            [f.size(0) for f in feats_list], dtype=torch.int32,
        )

        if len(feats_list) == 1:
            feats_pad = feats_list[0].unsqueeze(0)
        else:
            feats_pad = pad_sequence(
                feats_list, batch_first=True, padding_value=0.0,
            )

        if self.device.startswith("cuda"):
            feats_pad = feats_pad.pin_memory()

        return feats_pad, feats_lens

    def transcribe_preprocessed(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        *,
        language: str = "",
    ) -> list[ASRResult]:
        """Run inference directly on pre-computed fbank features.

        Bypasses AutoModel.inference() and its per-batch overheads
        (random key generation, torch.cuda.empty_cache, etc.).
        Automatically reduces chunk size on CUDA OOM.
        """
        B = feats.size(0)
        if B == 0:
            return []

        results: list[ASRResult] = []
        chunk_size = B
        offset = 0

        while offset < B:
            end = min(offset + chunk_size, B)
            try:
                chunk_results = self._encode_and_decode(
                    feats[offset:end],
                    feat_lengths[offset:end],
                    language=language,
                )
                results.extend(chunk_results)
                offset = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)

        return results

    def _encode_and_decode(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        *,
        language: str = "",
    ) -> list[ASRResult]:
        """Encode one chunk and CTC-decode the output."""
        effective_language = language.strip() or self.default_language
        sv = self._sensevoice
        dev = self.device

        speech = feats.to(device=dev, non_blocking=True)
        speech_lengths = feat_lengths.to(device=dev, non_blocking=True)

        lid = self._lid_dict.get(effective_language, 0)
        language_query = sv.embed(
            torch.LongTensor([[lid]]).to(dev)
        ).expand(speech.size(0), -1, -1)

        textnorm_key = "withitn" if self.use_itn else "woitn"
        textnorm_query = sv.embed(
            torch.LongTensor([[self._textnorm_dict[textnorm_key]]]).to(dev)
        ).expand(speech.size(0), -1, -1)

        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths = speech_lengths + 1

        event_emo_query = sv.embed(
            torch.LongTensor([[1, 2]]).to(dev)
        ).expand(speech.size(0), -1, -1)

        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths = speech_lengths + 3

        with torch.no_grad():
            encoder_out, encoder_out_lens = sv.encoder(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            ctc_logits = sv.ctc.log_softmax(encoder_out)
            all_preds = ctc_logits.argmax(dim=-1)

        all_preds_cpu = all_preds.cpu()
        encoder_out_lens_cpu = encoder_out_lens.cpu()

        del speech, speech_lengths, encoder_out, encoder_out_lens
        del ctc_logits, all_preds
        if dev != "cpu":
            torch.cuda.empty_cache()

        tokenizer = self._tokenizer
        blank_id = self._blank_id
        results: list[ASRResult] = []
        for i in range(all_preds_cpu.size(0)):
            length = encoder_out_lens_cpu[i].item()
            preds = all_preds_cpu[i, :length]
            yseq = torch.unique_consecutive(preds)
            token_int = yseq[yseq != blank_id].tolist()
            text = tokenizer.decode(token_int)
            results.append(ASRResult(text=text))

        return results

    @staticmethod
    def _normalize_result_items(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            items: list[Any] = [raw]
        elif isinstance(raw, list):
            items = raw
        elif isinstance(raw, tuple):
            items = list(raw)
        else:
            items = [raw]

        texts: list[str] = []
        for item in items:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                texts.append(str(getattr(item, "text")))
            else:
                texts.append(str(item))
        return texts

    def transcribe(
        self,
        *,
        audio: list[tuple[Any, int]],
        context: str = "",
        language: str = "",
    ) -> list[ASRResult]:
        del context
        if not audio:
            return []
        waveforms = [samples for samples, _sample_rate in audio]
        effective_language = language.strip() or self.default_language
        batch_size = min(self.max_inference_batch_size, len(waveforms))
        raw = self.model.generate(
            input=waveforms,
            cache={},
            language=effective_language,
            use_itn=self.use_itn,
            batch_size=batch_size,
        )
        texts = self._normalize_result_items(raw)
        if len(texts) != len(waveforms):
            raise RuntimeError(
                f"SenseVoice result size mismatch: got {len(texts)}, expected {len(waveforms)}"
            )
        return [ASRResult(text=text) for text in texts]


def _sensevoice_candidates(
    model_name: str,
    hub: str,
) -> list[tuple[str, str]]:
    model_name = model_name.strip()
    resolved_hub = hub.strip().lower()
    if resolved_hub in {"ms", "hf"}:
        return [(model_name, resolved_hub)]
    if resolved_hub != "auto":
        raise ValueError(f"Unsupported SenseVoice hub: {hub!r}. Use auto/ms/hf.")

    candidates: list[tuple[str, str]] = [
        (model_name, "ms"),
        (model_name, "hf"),
    ]
    if model_name == "iic/SenseVoiceSmall":
        candidates.append(("FunAudioLLM/SenseVoiceSmall", "hf"))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_asr_model(
    config: TranscribeConfig,
    resolved_device: str,
    segment_batch_size: int,
) -> Any:
    engine = config.asr_engine.strip().lower()
    if engine == "qwen3":
        return build_asr_model_qwen_vllm(
            config,
            resolved_device,
            segment_batch_size,
        )
    if engine != "sensevoice":
        raise ValueError(f"Unsupported ASR engine: {config.asr_engine}")
    return build_asr_model_sensevoice(
        config,
        resolved_device,
        segment_batch_size,
    )


def build_asr_model_sensevoice(
    config: TranscribeConfig,
    resolved_device: str,
    segment_batch_size: int,
) -> SenseVoiceASRModel:
    last_error: Exception | None = None
    candidates = _sensevoice_candidates(config.asr_model, config.asr_model_hub)
    for model_name, hub in candidates:
        try:
            model = AutoModel(
                model=model_name,
                hub=hub,
                device=resolved_device,
                disable_update=True,
                disable_pbar=True,
            )
            return SenseVoiceASRModel(
                model,
                default_language=config.asr_language,
                use_itn=config.asr_use_itn,
                max_inference_batch_size=segment_batch_size,
                device=resolved_device,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is None:
        raise RuntimeError("Failed to load SenseVoice model.")
    raise RuntimeError(
        "Failed to load SenseVoice model from all candidate hubs. "
        f"Last error: {last_error}"
    ) from last_error


def build_vad_model(
    device: str,
    vad_max_segment_ms: int,
    vad_max_end_silence_ms: int,
) -> Any:
    return AutoModel(
        model="fsmn-vad",
        hub="ms",
        device=device,
        disable_update=True,
        disable_pbar=True,
        max_single_segment_time=vad_max_segment_ms,
        max_end_silence_time=vad_max_end_silence_ms,
    )


class Qwen3VLLMFastWrapper:
    """Wraps Qwen3ASRModel to skip redundant audio normalization and prompt
    construction.  The pipeline already provides 16 kHz mono float32 segments,
    and all segments share the same context/language, so the prompt template
    is built once and reused.
    """

    def __init__(self, inner: Any, *, context: str, language: str) -> None:
        self.inner = inner
        self._cached_prompt: str | None = None
        self._cache_key: tuple[str, str | None] = ("", None)
        self._context = context
        self._language = language

    def _resolve_prompt(self, context: str, language: str | None) -> str:
        key = (context, language)
        if key == self._cache_key and self._cached_prompt is not None:
            return self._cached_prompt
        prompt = self.inner._build_text_prompt(
            context=context, force_language=language,
        )
        self._cached_prompt = prompt
        self._cache_key = key
        return prompt

    def transcribe(
        self,
        *,
        audio: list[tuple[Any, int]],
        context: str = "",
        language: str = "",
    ) -> list[ASRResult]:
        if not audio:
            return []

        effective_language = language.strip() or self._language or None
        effective_context = context.strip() or self._context

        prompt = self._resolve_prompt(effective_context, effective_language)

        from qwen_asr.inference.utils import (
            detect_and_fix_repetitions,
            parse_asr_output,
        )

        waveforms = [samples for samples, _sr in audio]
        inputs = [
            {"prompt": prompt, "multi_modal_data": {"audio": [w]}}
            for w in waveforms
        ]

        batch_size = self.inner.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(inputs)

        results: list[ASRResult] = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            outputs = self.inner.model.generate(
                batch,
                sampling_params=self.inner.sampling_params,
                use_tqdm=False,
            )
            for o in outputs:
                raw = o.outputs[0].text
                _lang, txt = parse_asr_output(raw, user_language=effective_language)
                results.append(ASRResult(text=txt))
        return results


def build_asr_model_qwen_vllm(
    config: TranscribeConfig,
    resolved_device: str,
    segment_batch_size: int,
) -> Qwen3VLLMFastWrapper:
    if not resolved_device.startswith("cuda"):
        raise RuntimeError(
            f"Qwen3 vLLM requires CUDA, got resolved device {resolved_device!r}. "
            "Use --asr-engine sensevoice on CPU."
        )
    resolve_vllm_device(resolved_device)
    prepare_qwen_runtime(config.qwen_src_dir, config.qwen_repo_url)
    try:
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import Qwen3ASRModel for vLLM backend. "
            "Install dependencies with `pip install -U qwen-asr[vllm] vllm`."
        ) from exc

    inner = Qwen3ASRModel.LLM(
        model=config.qwen_model,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        max_inference_batch_size=segment_batch_size,
        max_new_tokens=config.qwen_max_new_tokens,
        max_model_len=config.vllm_max_model_len,
        max_num_seqs=config.vllm_max_num_seqs,
        disable_log_stats=True,
    )

    context = config.qwen_context.strip() if config.use_prompt else ""
    from qwen_asr.inference.utils import normalize_language_name

    language = normalize_language_name(config.qwen_language)
    return Qwen3VLLMFastWrapper(inner, context=context, language=language)
