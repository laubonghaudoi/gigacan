from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.config import TranscribeConfig
from gigacan.qwen_srt.worker_client import _runtime_signature, _worker_command


def _config() -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("a.opus"),
        output_srt=Path("a.srt"),
        asr_backend="vllm",
        vad_device="cuda",
        vllm_gpu_memory_utilization=0.8,
        vllm_tensor_parallel_size=2,
    )


def test_runtime_signature_includes_backend_fields() -> None:
    signature = _runtime_signature(_config())
    assert signature["asr_backend"] == "vllm"
    assert signature["vad_device"] == "cuda"
    assert signature["vllm_gpu_memory_utilization"] == 0.8
    assert signature["vllm_tensor_parallel_size"] == 2


def test_worker_command_includes_backend_fields() -> None:
    cmd = _worker_command(_config(), Path("/tmp/qwen_srt.sock"))
    assert "--asr-backend" in cmd
    asr_idx = cmd.index("--asr-backend")
    assert cmd[asr_idx + 1] == "vllm"
    vad_dev_idx = cmd.index("--vad-device")
    assert cmd[vad_dev_idx + 1] == "cuda"

    gpu_idx = cmd.index("--vllm-gpu-memory-utilization")
    tp_idx = cmd.index("--vllm-tensor-parallel-size")
    assert cmd[gpu_idx + 1] == "0.8"
    assert cmd[tp_idx + 1] == "2"
