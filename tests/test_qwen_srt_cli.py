from __future__ import annotations

from pathlib import Path

import pytest

from gigacan.qwen_srt.cli import build_config, parse_args
from gigacan.qwen_srt.config import (
    DEFAULT_ASR_PREFETCH_BATCHES,
    DEFAULT_PREP_WORKERS,
    DEFAULT_SEGMENT_BATCH_SIZE,
    DEFAULT_SUPER_BATCH_ACTIVE_FILES,
    DEFAULT_SUPER_BATCH_MAX_DECODED_GIB,
    DEFAULT_SUPER_BATCH_PRELOAD_FILES,
    DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER,
)


def test_parse_args_defaults_to_sensevoice_backend() -> None:
    ns = parse_args([])
    assert ns.asr_backend == "sensevoice"
    assert ns.asr_model == "iic/SenseVoiceSmall"
    assert ns.asr_model_hub == "auto"
    assert ns.asr_language == "yue"
    assert ns.asr_use_itn is True
    assert ns.vad_device == "auto"
    assert ns.vad_max_segment_ms == 15000
    assert ns.vad_max_end_silence_ms == 500
    assert ns.segment_batch_size == DEFAULT_SEGMENT_BATCH_SIZE
    assert ns.super_batch_active_files == DEFAULT_SUPER_BATCH_ACTIVE_FILES
    assert ns.super_batch_queue_multiplier == DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER
    assert ns.super_batch_preload_files == DEFAULT_SUPER_BATCH_PRELOAD_FILES
    assert ns.super_batch_max_decoded_gib == DEFAULT_SUPER_BATCH_MAX_DECODED_GIB
    assert ns.prep_workers == DEFAULT_PREP_WORKERS
    assert ns.asr_prefetch_batches == DEFAULT_ASR_PREFETCH_BATCHES


def test_parse_args_rejects_vllm_backend() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--asr-backend", "vllm"])


def test_parse_args_rejects_empty_asr_model() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--asr-model", "   "])


def test_parse_args_rejects_negative_vad_max_end_silence_ms() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--vad-max-end-silence-ms", "-1"])


def test_build_config_maps_backend_fields() -> None:
    ns = parse_args(
        [
            "--asr-backend",
            "transformers",
            "--vad-device",
            "cuda",
            "--asr-model",
            "FunAudioLLM/SenseVoiceSmall",
            "--asr-model-hub",
            "hf",
            "--asr-language",
            "yue",
            "--no-asr-use-itn",
        ]
    )
    config = build_config(ns, audio=Path("a.opus"), output_srt=Path("a.srt"))
    assert config.asr_backend == "sensevoice"
    assert config.vad_device == "cuda"
    assert config.asr_model == "FunAudioLLM/SenseVoiceSmall"
    assert config.asr_model_hub == "hf"
    assert config.asr_language == "yue"
    assert config.asr_use_itn is False
