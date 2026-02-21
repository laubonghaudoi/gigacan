from __future__ import annotations

from pathlib import Path

import pytest

from gigacan.qwen_srt.cli import build_config, parse_args
from gigacan.qwen_srt.config import (
    DEFAULT_ASR_ENGINE,
    DEFAULT_ASR_PREFETCH_BATCHES,
    DEFAULT_PREP_WORKERS,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SEGMENT_BATCH_SIZE,
    DEFAULT_SUPER_BATCH_ACTIVE_FILES,
    DEFAULT_SUPER_BATCH_MAX_DECODED_GIB,
    DEFAULT_SUPER_BATCH_PRELOAD_FILES,
    DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER,
    DEFAULT_VAD_MAX_END_SILENCE_MS,
    DEFAULT_VAD_MAX_SEGMENT_MS,
    DEFAULT_VAD_WORKERS,
)


def test_parse_args_defaults_to_qwen3_engine() -> None:
    ns = parse_args([])
    assert ns.asr_engine == DEFAULT_ASR_ENGINE
    assert ns.segment_batch_size == DEFAULT_SEGMENT_BATCH_SIZE
    assert ns.super_batch_active_files == DEFAULT_SUPER_BATCH_ACTIVE_FILES
    assert ns.super_batch_queue_multiplier == DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER
    assert ns.super_batch_preload_files == DEFAULT_SUPER_BATCH_PRELOAD_FILES
    assert ns.super_batch_max_decoded_gib == DEFAULT_SUPER_BATCH_MAX_DECODED_GIB
    assert ns.vad_max_segment_ms == DEFAULT_VAD_MAX_SEGMENT_MS
    assert ns.prep_workers == DEFAULT_PREP_WORKERS
    assert ns.vad_workers == DEFAULT_VAD_WORKERS
    assert ns.asr_prefetch_batches == DEFAULT_ASR_PREFETCH_BATCHES
    assert ns.vllm_gpu_memory_utilization == 0.9
    assert ns.vllm_tensor_parallel_size == 1


def test_parse_args_uses_same_common_defaults_for_sensevoice() -> None:
    qwen = parse_args([])
    sv = parse_args(["--asr-engine", "sensevoice"])
    assert sv.segment_batch_size == qwen.segment_batch_size
    assert sv.super_batch_active_files == qwen.super_batch_active_files
    assert sv.super_batch_queue_multiplier == qwen.super_batch_queue_multiplier
    assert sv.super_batch_preload_files == qwen.super_batch_preload_files
    assert sv.super_batch_max_decoded_gib == qwen.super_batch_max_decoded_gib
    assert sv.vad_max_segment_ms == qwen.vad_max_segment_ms
    assert sv.prep_workers == qwen.prep_workers
    assert sv.vad_workers == qwen.vad_workers
    assert sv.asr_prefetch_batches == qwen.asr_prefetch_batches


def test_parse_args_rejects_legacy_asr_backend_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--asr-backend", "vllm"])


def test_parse_args_rejects_removed_vad_max_end_silence_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vad-max-end-silence-ms", "-1"])


def test_parse_args_rejects_invalid_vllm_tensor_parallel_size() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vllm-tensor-parallel-size", "0"])


def test_parse_args_rejects_invalid_vllm_gpu_memory_utilization() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vllm-gpu-memory-utilization", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--vllm-gpu-memory-utilization", "1.1"])


def test_build_config_maps_qwen_fields() -> None:
    ns = parse_args(
        [
            "--asr-engine",
            "qwen3",
            "--qwen-language",
            "Cantonese",
            "--qwen-context",
            "meeting context",
            "--use-prompt",
            "--vllm-gpu-memory-utilization",
            "0.85",
            "--vllm-tensor-parallel-size",
            "2",
            "--qwen-max-new-tokens",
            "320",
        ]
    )
    config = build_config(ns, audio=Path("a.opus"), output_srt=Path("a.srt"))
    assert config.asr_engine == "qwen3"
    assert config.qwen_model == DEFAULT_QWEN_MODEL
    assert config.qwen_language == "Cantonese"
    assert config.qwen_context == "meeting context"
    assert config.use_prompt is True
    assert config.vllm_gpu_memory_utilization == 0.85
    assert config.vllm_tensor_parallel_size == 2
    assert config.qwen_max_new_tokens == 320
    assert config.vad_max_end_silence_ms == DEFAULT_VAD_MAX_END_SILENCE_MS


def test_build_config_maps_sensevoice_fields() -> None:
    ns = parse_args(
        [
            "--asr-engine",
            "sensevoice",
            "--asr-model-hub",
            "hf",
            "--asr-language",
            "yue",
            "--no-asr-use-itn",
        ]
    )
    config = build_config(ns, audio=Path("a.opus"), output_srt=Path("a.srt"))
    assert config.asr_engine == "sensevoice"
    assert config.asr_model_hub == "hf"
    assert config.asr_language == "yue"
    assert config.asr_use_itn is False
    assert config.vad_max_end_silence_ms == DEFAULT_VAD_MAX_END_SILENCE_MS
