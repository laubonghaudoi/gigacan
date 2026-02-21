from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.config import (
    DEFAULT_VAD_MAX_END_SILENCE_MS,
    TranscribeConfig,
)
from gigacan.qwen_srt.worker_client import _runtime_signature, _worker_command


def _sensevoice_config() -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_engine="sensevoice",
        asr_model_hub="ms",
        asr_language="yue",
        asr_use_itn=False,
        vad_device="cuda",
    )


def _qwen_config() -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_engine="qwen3",
        qwen_language="Cantonese",
        qwen_context="meeting context",
        use_prompt=True,
        vllm_gpu_memory_utilization=0.8,
        vllm_tensor_parallel_size=2,
        qwen_max_new_tokens=320,
    )


def test_runtime_signature_includes_engine_fields() -> None:
    signature = _runtime_signature(_sensevoice_config())
    assert signature["asr_engine"] == "sensevoice"
    assert signature["vad_device"] == "cuda"
    assert signature["vad_max_end_silence_ms"] == DEFAULT_VAD_MAX_END_SILENCE_MS
    assert signature["asr_model_hub"] == "ms"
    assert signature["asr_language"] == "yue"
    assert signature["asr_use_itn"] is False


def test_worker_command_includes_sensevoice_fields() -> None:
    cmd = _worker_command(_sensevoice_config(), Path("/tmp/qwen_srt.sock"))
    assert "--asr-engine" in cmd
    asr_idx = cmd.index("--asr-engine")
    assert cmd[asr_idx + 1] == "sensevoice"
    vad_dev_idx = cmd.index("--vad-device")
    assert cmd[vad_dev_idx + 1] == "cuda"
    end_sil_idx = cmd.index("--vad-max-end-silence-ms")
    assert cmd[end_sil_idx + 1] == str(DEFAULT_VAD_MAX_END_SILENCE_MS)
    hub_idx = cmd.index("--asr-model-hub")
    lang_idx = cmd.index("--asr-language")
    assert cmd[hub_idx + 1] == "ms"
    assert cmd[lang_idx + 1] == "yue"
    assert "--no-asr-use-itn" in cmd


def test_runtime_signature_includes_qwen_vllm_fields() -> None:
    signature = _runtime_signature(_qwen_config())
    assert signature["asr_engine"] == "qwen3"
    assert signature["qwen_language"] == "Cantonese"
    assert signature["qwen_context"] == "meeting context"
    assert signature["use_prompt"] is True
    assert signature["vllm_gpu_memory_utilization"] == 0.8
    assert signature["vllm_tensor_parallel_size"] == 2
    assert signature["qwen_max_new_tokens"] == 320


def test_worker_command_includes_qwen_vllm_fields() -> None:
    cmd = _worker_command(_qwen_config(), Path("/tmp/qwen_srt.sock"))
    asr_idx = cmd.index("--asr-engine")
    gpu_idx = cmd.index("--vllm-gpu-memory-utilization")
    tp_idx = cmd.index("--vllm-tensor-parallel-size")
    lang_idx = cmd.index("--qwen-language")
    ctx_idx = cmd.index("--qwen-context")
    tok_idx = cmd.index("--qwen-max-new-tokens")
    assert cmd[asr_idx + 1] == "qwen3"
    assert cmd[gpu_idx + 1] == "0.8"
    assert cmd[tp_idx + 1] == "2"
    assert cmd[lang_idx + 1] == "Cantonese"
    assert cmd[ctx_idx + 1] == "meeting context"
    assert cmd[tok_idx + 1] == "320"
    assert "--use-prompt" in cmd
