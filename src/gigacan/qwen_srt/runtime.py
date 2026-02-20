from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from funasr import AutoModel

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
    ) -> None:
        self.model = model
        self.default_language = default_language
        self.use_itn = use_itn
        self.max_inference_batch_size = max(1, int(max_inference_batch_size))

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
    backend = config.asr_backend.strip().lower()
    if backend == "vllm":
        raise RuntimeError(
            "SenseVoice backend does not support vLLM. "
            "Use --asr-backend sensevoice (or transformers alias)."
        )
    if backend not in {"sensevoice", "transformers"}:
        raise ValueError(f"Unsupported ASR backend: {config.asr_backend}")
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
