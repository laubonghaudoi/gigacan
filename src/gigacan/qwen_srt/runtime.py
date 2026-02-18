from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import torch
from funasr import AutoModel
from transformers import AutoModel as HFAutoModel
from transformers import AutoProcessor

from .config import TranscribeConfig


def resolve_backend(backend_arg: str) -> str:
    backend = backend_arg.strip().lower()
    if backend not in {"vllm", "transformers"}:
        raise ValueError(
            f"Unsupported backend: {backend_arg}. Expected one of: vllm, transformers."
        )
    return backend


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_segment_batch_size(
    device: str,
    batch_size_arg: int,
    *,
    backend: str,
) -> int:
    if batch_size_arg > 0:
        return batch_size_arg
    if device.startswith("cuda"):
        return 256 if backend == "vllm" else 128
    return 4


def resolve_qwen_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def check_vllm_prerequisites() -> tuple[bool, str]:
    include_dir = Path(sysconfig.get_paths().get("include", ""))
    header = include_dir / "Python.h"
    if not header.is_file():
        return (
            False,
            f"Missing Python headers ({header}). Install python3-dev / python3.12-dev.",
        )
    return True, ""


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


def ensure_qwen_config_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRThinkerConfig,
        )
    except Exception:
        return

    if not hasattr(Qwen3ASRThinkerConfig, "pad_token_id"):
        Qwen3ASRThinkerConfig.pad_token_id = -1  # type: ignore[attr-defined]


def ensure_qwen_modeling_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding,
        )
    except Exception:
        return

    if hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
        return

    def compute_default_rope_parameters(
        self: Any,
        config: Any,
        device: Any = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len, layer_type
        rope_init_fn = getattr(self, "rope_init_fn", None)
        if rope_init_fn is None:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
        return rope_init_fn(config, device)

    Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = (  # type: ignore[attr-defined]
        compute_default_rope_parameters
    )


def move_qwen_model_to_device(qwen_model: Any, device: str) -> None:
    if not device.startswith("cuda"):
        return
    qwen_model.model = qwen_model.model.to(device)
    qwen_model.device = next(qwen_model.model.parameters()).device
    qwen_model.dtype = qwen_model.model.dtype


def prepare_qwen_runtime(
    src_dir: Path,
    repo_url: str,
    *,
    enable_transformers_compat: bool,
) -> None:
    qwen_src_dir = ensure_qwen_source(src_dir, repo_url)
    if str(qwen_src_dir) not in sys.path:
        sys.path.insert(0, str(qwen_src_dir))
    if enable_transformers_compat:
        ensure_qwen_transformers_compat()


def build_asr_model(
    config: TranscribeConfig,
    resolved_device: str,
    qwen_dtype: torch.dtype | str,
    segment_batch_size: int,
) -> Any:
    backend = resolve_backend(config.backend)
    prepare_qwen_runtime(
        config.qwen_src_dir,
        config.qwen_repo_url,
        enable_transformers_compat=backend == "transformers",
    )

    from qwen_asr import Qwen3ASRModel

    if backend == "vllm":
        if not resolved_device.startswith("cuda"):
            raise ValueError(
                'The "vllm" backend requires CUDA. Use --device cuda:0 or switch to --backend transformers.'
            )
        ready, reason = check_vllm_prerequisites()
        if ready:
            vllm_kwargs: dict[str, Any] = {
                "gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            }
            if config.vllm_max_num_seqs > 0:
                vllm_kwargs["max_num_seqs"] = config.vllm_max_num_seqs
            if config.vllm_max_num_batched_tokens > 0:
                vllm_kwargs["max_num_batched_tokens"] = (
                    config.vllm_max_num_batched_tokens
                )
            if config.vllm_max_model_len > 0:
                vllm_kwargs["max_model_len"] = config.vllm_max_model_len
            if config.qwen_dtype != "auto":
                vllm_kwargs["dtype"] = config.qwen_dtype
            try:
                llm_model = Qwen3ASRModel.LLM(
                    model=config.qwen_model,
                    max_inference_batch_size=segment_batch_size,
                    max_new_tokens=config.qwen_max_new_tokens,
                    **vllm_kwargs,
                )
                setattr(llm_model, "_gigacan_backend", "vllm")
                return llm_model
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM backend requested but not available. Install with `pip install qwen-asr[vllm]` or `pip install vllm`."
                ) from exc
            except Exception as exc:  # noqa: BLE001
                print(
                    "Warning: vLLM backend initialization failed; falling back to transformers backend."
                )
                print(f"vLLM error: {exc}")
                print(
                    "Hint: install Python development headers (python3-dev / python3.12-dev) if you want vLLM."
                )
        else:
            print(
                "Warning: vLLM prerequisites are not met; falling back to transformers backend."
            )
            print(f"Reason: {reason}")
        backend = "transformers"
        qwen_dtype = resolve_qwen_dtype(config.qwen_dtype, resolved_device)

    if not isinstance(qwen_dtype, torch.dtype):
        raise TypeError(f"Expected torch.dtype for transformers backend, got {qwen_dtype}")

    ensure_qwen_config_compat()
    ensure_qwen_modeling_compat()
    model = HFAutoModel.from_pretrained(
        config.qwen_model,
        dtype=qwen_dtype,
    )
    try:
        processor = AutoProcessor.from_pretrained(
            config.qwen_model, fix_mistral_regex=True
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(config.qwen_model)

    asr_model = Qwen3ASRModel(
        backend="transformers",
        model=model,
        processor=processor,
        sampling_params=None,
        forced_aligner=None,
        max_inference_batch_size=segment_batch_size,
        max_new_tokens=config.qwen_max_new_tokens,
    )
    move_qwen_model_to_device(asr_model, resolved_device)
    setattr(asr_model, "_gigacan_backend", "transformers")
    if hasattr(asr_model.model, "generation_config"):
        eos_token_id = getattr(asr_model.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            asr_model.model.generation_config.pad_token_id = eos_token_id[-1]
    return asr_model


def build_vad_model(device: str, vad_max_segment_ms: int) -> Any:
    return AutoModel(
        model="fsmn-vad",
        hub="ms",
        device=device,
        disable_update=True,
        max_single_segment_time=vad_max_segment_ms,
    )
