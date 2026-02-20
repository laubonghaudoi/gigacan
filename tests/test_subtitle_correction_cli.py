from __future__ import annotations

import pytest

from gigacan.subtitle_correction.cli import parse_args


def test_parse_args_defaults_to_vllm_backend() -> None:
    ns = parse_args(["--year", "2025"])
    assert ns.backend == "vllm"
    assert ns.ollama_host == "http://127.0.0.1:11434"


def test_parse_args_accepts_ollama_backend() -> None:
    ns = parse_args(
        [
            "--year",
            "2025",
            "--backend",
            "ollama",
            "--model",
            "gemma3:27b",
            "--ollama-host",
            "http://localhost:11434",
        ]
    )
    assert ns.backend == "ollama"
    assert ns.model == "gemma3:27b"
    assert ns.ollama_host == "http://localhost:11434"


def test_parse_args_rejects_invalid_vllm_tensor_parallel_size() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--year", "2025", "--vllm-tensor-parallel-size", "0"])
