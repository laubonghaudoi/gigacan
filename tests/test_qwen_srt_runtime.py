from __future__ import annotations

from pathlib import Path

import pytest

from gigacan.qwen_srt import runtime
from gigacan.qwen_srt.config import TranscribeConfig


def _config(*, backend: str) -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_backend=backend,
    )


def test_build_asr_model_dispatches_to_sensevoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(backend="sensevoice")
    called: dict[str, int] = {"sensevoice": 0}

    def fake_sensevoice(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["sensevoice"] += 1
        return "sensevoice-model"

    monkeypatch.setattr(runtime, "build_asr_model_sensevoice", fake_sensevoice)
    result = runtime.build_asr_model(config, "cuda:0", 32)
    assert result == "sensevoice-model"
    assert called == {"sensevoice": 1}


def test_build_asr_model_allows_transformers_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(backend="transformers")
    called: dict[str, int] = {"sensevoice": 0}

    def fake_sensevoice(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["sensevoice"] += 1
        return "sensevoice-model"

    monkeypatch.setattr(runtime, "build_asr_model_sensevoice", fake_sensevoice)
    result = runtime.build_asr_model(config, "cuda:0", 32)
    assert result == "sensevoice-model"
    assert called == {"sensevoice": 1}


def test_build_asr_model_rejects_vllm() -> None:
    config = _config(backend="vllm")
    with pytest.raises(RuntimeError):
        runtime.build_asr_model(config, "cuda:0", 32)


def test_build_asr_model_rejects_unknown_backend() -> None:
    config = _config(backend="unknown")
    with pytest.raises(ValueError):
        runtime.build_asr_model(config, "cuda:0", 32)


def test_sensevoice_candidates_auto_includes_ms_and_hf() -> None:
    candidates = runtime._sensevoice_candidates("iic/SenseVoiceSmall", "auto")
    assert ("iic/SenseVoiceSmall", "ms") in candidates
    assert ("iic/SenseVoiceSmall", "hf") in candidates
    assert ("FunAudioLLM/SenseVoiceSmall", "hf") in candidates


def test_sensevoice_wrapper_normalizes_text_outputs() -> None:
    class FakeAutoModel:
        def generate(self, **_kwargs):
            return [
                {"text": "first"},
                "second",
            ]

    wrapper = runtime.SenseVoiceASRModel(
        FakeAutoModel(),
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
    )
    results = wrapper.transcribe(
        audio=[([0.0, 0.1], 16000), ([0.2, 0.3], 16000)],
    )
    assert [item.text for item in results] == ["first", "second"]
