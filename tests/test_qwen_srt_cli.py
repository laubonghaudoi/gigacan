from __future__ import annotations

from pathlib import Path

import pytest

from gigacan.qwen_srt.cli import build_config, parse_args


def test_parse_args_defaults_to_vllm_backend() -> None:
    ns = parse_args([])
    assert ns.asr_backend == "vllm"
    assert ns.vllm_gpu_memory_utilization == 0.7
    assert ns.vllm_tensor_parallel_size == 1


def test_parse_args_rejects_invalid_vllm_tensor_parallel_size() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vllm-tensor-parallel-size", "0"])


def test_parse_args_rejects_invalid_vllm_gpu_memory_utilization() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vllm-gpu-memory-utilization", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--vllm-gpu-memory-utilization", "1.1"])


def test_build_config_maps_backend_fields() -> None:
    ns = parse_args(
        [
            "--asr-backend",
            "transformers",
            "--vllm-gpu-memory-utilization",
            "0.85",
            "--vllm-tensor-parallel-size",
            "2",
        ]
    )
    config = build_config(ns, audio=Path("a.opus"), output_srt=Path("a.srt"))
    assert config.asr_backend == "transformers"
    assert config.vllm_gpu_memory_utilization == 0.85
    assert config.vllm_tensor_parallel_size == 2
