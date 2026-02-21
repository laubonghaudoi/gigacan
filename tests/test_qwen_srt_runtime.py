from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gigacan.qwen_srt import runtime
from gigacan.qwen_srt.config import TranscribeConfig
from gigacan.qwen_srt.runtime import _apply_cmvn, _apply_lfr


def _config(*, engine: str) -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_engine=engine,
    )


def test_build_asr_model_dispatches_to_sensevoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(engine="sensevoice")
    called: dict[str, int] = {"sensevoice": 0, "qwen_vllm": 0}

    def fake_sensevoice(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["sensevoice"] += 1
        return "sensevoice-model"

    def fake_qwen_vllm(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["qwen_vllm"] += 1
        return "qwen-vllm-model"

    monkeypatch.setattr(runtime, "build_asr_model_sensevoice", fake_sensevoice)
    monkeypatch.setattr(runtime, "build_asr_model_qwen_vllm", fake_qwen_vllm)
    result = runtime.build_asr_model(config, "cuda:0", 32)
    assert result == "sensevoice-model"
    assert called == {"sensevoice": 1, "qwen_vllm": 0}


def test_build_asr_model_dispatches_to_qwen_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(engine="qwen3")
    called: dict[str, int] = {"sensevoice": 0, "qwen_vllm": 0}

    def fake_sensevoice(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["sensevoice"] += 1
        return "sensevoice-model"

    def fake_qwen_vllm(
        _config: TranscribeConfig,
        _resolved_device: str,
        _segment_batch_size: int,
    ) -> str:
        called["qwen_vllm"] += 1
        return "qwen-vllm-model"

    monkeypatch.setattr(runtime, "build_asr_model_sensevoice", fake_sensevoice)
    monkeypatch.setattr(runtime, "build_asr_model_qwen_vllm", fake_qwen_vllm)
    result = runtime.build_asr_model(config, "cuda:0", 32)
    assert result == "qwen-vllm-model"
    assert called == {"sensevoice": 0, "qwen_vllm": 1}


def test_build_asr_model_rejects_unknown_engine() -> None:
    config = _config(engine="unknown")
    with pytest.raises(ValueError):
        runtime.build_asr_model(config, "cuda:0", 32)


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


def test_apply_lfr_reduces_frame_count() -> None:
    """LFR with m=7, n=6 stacks 7 frames and subsamples by 6."""
    T, D = 60, 80
    x = torch.randn(T, D)
    out = _apply_lfr(x, lfr_m=7, lfr_n=6)
    expected_T = int(np.ceil(T / 6))
    assert out.shape == (expected_T, D * 7)
    assert out.dtype == torch.float32


def test_apply_lfr_identity_when_1_1() -> None:
    x = torch.randn(10, 80)
    out = _apply_lfr(x, lfr_m=1, lfr_n=1)
    assert out.shape == x.shape
    assert torch.allclose(out, x.to(torch.float32))


def test_apply_cmvn_normalizes_features() -> None:
    feats = torch.ones(5, 4, dtype=torch.float32)
    cmvn = torch.tensor([
        [-1.0, -1.0, -1.0, -1.0],  # means (additive shift)
        [2.0, 2.0, 2.0, 2.0],      # vars (multiplicative scale)
    ])
    out = _apply_cmvn(feats.clone(), cmvn)
    # (1 + (-1)) * 2 = 0
    assert torch.allclose(out, torch.zeros(5, 4))


def _make_fake_automodel_with_internals() -> SimpleNamespace:
    """Build a fake AutoModel with enough internals for direct inference."""
    fake_frontend = SimpleNamespace(
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        window="hamming",
        fs=16000,
        snip_edges=True,
        upsacle_samples=True,
        lfr_m=1,
        lfr_n=1,
        cmvn=None,
    )

    fake_tokenizer = MagicMock()
    fake_tokenizer.decode.return_value = "hello world"

    fake_sensevoice = SimpleNamespace(
        blank_id=0,
        lid_dict={"auto": 0, "yue": 7},
        textnorm_dict={"withitn": 14, "woitn": 15},
        embed=None,
        encoder=None,
        ctc=None,
    )

    return SimpleNamespace(
        model=fake_sensevoice,
        kwargs={
            "frontend": fake_frontend,
            "tokenizer": fake_tokenizer,
        },
        generate=lambda **kw: [{"text": "fallback"}],
    )


def test_direct_ready_set_when_internals_present() -> None:
    fake = _make_fake_automodel_with_internals()
    wrapper = runtime.SenseVoiceASRModel(
        fake,
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
    )
    assert wrapper._direct_ready is True


def test_direct_ready_false_for_plain_mock() -> None:
    class PlainFake:
        def generate(self, **_kw):
            return []

    wrapper = runtime.SenseVoiceASRModel(
        PlainFake(),
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
    )
    assert wrapper._direct_ready is False


def test_extract_features_returns_padded_tensor() -> None:
    fake = _make_fake_automodel_with_internals()
    wrapper = runtime.SenseVoiceASRModel(
        fake,
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
        fbank_workers=2,
    )
    wf1 = np.random.randn(16000).astype(np.float32)  # 1 second
    wf2 = np.random.randn(8000).astype(np.float32)   # 0.5 seconds
    feats, lengths = wrapper.extract_features([wf1, wf2])

    assert feats.ndim == 3
    assert feats.shape[0] == 2
    assert feats.shape[2] == 80  # n_mels=80, lfr_m=1
    assert lengths.shape == (2,)
    assert lengths[0].item() > lengths[1].item()


def test_extract_features_empty_input() -> None:
    fake = _make_fake_automodel_with_internals()
    wrapper = runtime.SenseVoiceASRModel(
        fake,
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
    )
    feats, lengths = wrapper.extract_features([])
    assert feats.shape[0] == 0
    assert lengths.shape[0] == 0


def test_transcribe_preprocessed_returns_results() -> None:
    """Verify transcribe_preprocessed calls encoder and decodes CTC output."""
    fake = _make_fake_automodel_with_internals()
    FEAT_DIM = 560
    B, T_ENC, VOCAB = 2, 10, 100
    encoder_out = torch.randn(B, T_ENC, FEAT_DIM)
    encoder_out_lens = torch.tensor([T_ENC, T_ENC - 2])

    fake_encoder = MagicMock(return_value=(encoder_out, encoder_out_lens))
    fake.model.encoder = fake_encoder

    fake_embed = MagicMock()
    fake_embed.return_value = torch.randn(1, 1, FEAT_DIM)
    fake.model.embed = fake_embed

    fake_ctc = MagicMock()
    fake_ctc.log_softmax = MagicMock(return_value=torch.randn(B, T_ENC, VOCAB))
    fake.model.ctc = fake_ctc

    wrapper = runtime.SenseVoiceASRModel(
        fake,
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
        device="cpu",
    )

    feats = torch.randn(B, 20, FEAT_DIM)
    feat_lengths = torch.tensor([20, 15], dtype=torch.int32)
    results = wrapper.transcribe_preprocessed(feats, feat_lengths)

    assert len(results) == B
    assert all(isinstance(r, runtime.ASRResult) for r in results)
    assert all(r.text == "hello world" for r in results)
    fake_encoder.assert_called_once()


def test_transcribe_preprocessed_empty() -> None:
    fake = _make_fake_automodel_with_internals()
    wrapper = runtime.SenseVoiceASRModel(
        fake,
        default_language="yue",
        use_itn=True,
        max_inference_batch_size=8,
    )
    results = wrapper.transcribe_preprocessed(
        torch.zeros(0, 0, 0),
        torch.zeros(0, dtype=torch.int32),
    )
    assert results == []
