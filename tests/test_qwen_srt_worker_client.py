from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.config import TranscribeConfig
from gigacan.qwen_srt.worker_client import _runtime_signature, _worker_command


def _config() -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_backend="sensevoice",
        asr_model="iic/SenseVoiceSmall",
        asr_model_hub="ms",
        asr_language="yue",
        asr_use_itn=False,
        vad_device="cuda",
    )


def test_runtime_signature_includes_backend_fields() -> None:
    signature = _runtime_signature(_config())
    assert signature["asr_backend"] == "sensevoice"
    assert signature["vad_device"] == "cuda"
    assert signature["vad_max_end_silence_ms"] == 500
    assert signature["asr_model"] == "iic/SenseVoiceSmall"
    assert signature["asr_model_hub"] == "ms"
    assert signature["asr_language"] == "yue"
    assert signature["asr_use_itn"] is False


def test_worker_command_includes_backend_fields() -> None:
    cmd = _worker_command(_config(), Path("/tmp/qwen_srt.sock"))
    assert "--asr-backend" in cmd
    asr_idx = cmd.index("--asr-backend")
    assert cmd[asr_idx + 1] == "sensevoice"
    vad_dev_idx = cmd.index("--vad-device")
    assert cmd[vad_dev_idx + 1] == "cuda"
    end_sil_idx = cmd.index("--vad-max-end-silence-ms")
    assert cmd[end_sil_idx + 1] == "500"

    model_idx = cmd.index("--asr-model")
    hub_idx = cmd.index("--asr-model-hub")
    lang_idx = cmd.index("--asr-language")
    assert cmd[model_idx + 1] == "iic/SenseVoiceSmall"
    assert cmd[hub_idx + 1] == "ms"
    assert cmd[lang_idx + 1] == "yue"
    assert "--no-asr-use-itn" in cmd
