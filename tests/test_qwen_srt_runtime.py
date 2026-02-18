from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gigacan.qwen_srt import runtime
from gigacan.qwen_srt.config import TranscribeConfig


def _config(*, backend: str) -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_backend=backend,
    )


def test_build_asr_model_dispatches_to_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(backend="vllm")
    called: dict[str, int] = {"vllm": 0, "transformers": 0}

    def fake_vllm(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["vllm"] += 1
        return "vllm-model"

    def fake_transformers(*_args: object, **_kwargs: object) -> str:
        called["transformers"] += 1
        return "transformers-model"

    monkeypatch.setattr(runtime, "build_asr_model_vllm", fake_vllm)
    monkeypatch.setattr(runtime, "build_asr_model_transformers", fake_transformers)

    result = runtime.build_asr_model(config, "cuda:0", torch.bfloat16, 32)
    assert result == "vllm-model"
    assert called == {"vllm": 1, "transformers": 0}


def test_build_asr_model_dispatches_to_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(backend="transformers")
    called: dict[str, int] = {"vllm": 0, "transformers": 0}

    def fake_vllm(*_args: object, **_kwargs: object) -> str:
        called["vllm"] += 1
        return "vllm-model"

    def fake_transformers(
        _config: TranscribeConfig,
        _resolved_device: str,
        _qwen_dtype: torch.dtype,
        _segment_batch_size: int,
    ) -> str:
        called["transformers"] += 1
        return "transformers-model"

    monkeypatch.setattr(runtime, "build_asr_model_vllm", fake_vllm)
    monkeypatch.setattr(runtime, "build_asr_model_transformers", fake_transformers)

    result = runtime.build_asr_model(config, "cuda:0", torch.bfloat16, 32)
    assert result == "transformers-model"
    assert called == {"vllm": 0, "transformers": 1}


def test_build_asr_model_rejects_unknown_backend() -> None:
    config = _config(backend="unknown")
    with pytest.raises(ValueError):
        runtime.build_asr_model(config, "cuda:0", torch.bfloat16, 32)


def test_resolve_vllm_device_sets_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    resolved = runtime.resolve_vllm_device("cuda:2")
    assert resolved == "cuda:0"
    assert runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_resolve_vllm_device_rejects_conflicting_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError):
        runtime.resolve_vllm_device("cuda:2")


def test_resolve_vllm_device_rejects_cpu() -> None:
    with pytest.raises(RuntimeError):
        runtime.resolve_vllm_device("cpu")
