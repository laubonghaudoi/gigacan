from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ASR_MODEL = "iic/SenseVoiceSmall"
DEFAULT_ASR_LANGUAGE = "yue"
DEFAULT_SEGMENT_BATCH_SIZE = 1536
DEFAULT_SUPER_BATCH_ACTIVE_FILES = 48
DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER = 48
DEFAULT_SUPER_BATCH_PRELOAD_FILES = 96
DEFAULT_SUPER_BATCH_MAX_DECODED_GIB = 40.0
DEFAULT_PREP_WORKERS = 20
DEFAULT_VAD_WORKERS = 1
DEFAULT_ASR_PREFETCH_BATCHES = 24


@dataclass(slots=True)
class TranscribeConfig:
    """Configuration for one audio-to-SRT transcription run."""

    audio: Path
    output_srt: Path
    asr_backend: str = "sensevoice"
    asr_model: str = DEFAULT_ASR_MODEL
    asr_model_hub: str = "auto"
    asr_language: str = DEFAULT_ASR_LANGUAGE
    asr_use_itn: bool = True
    device: str = "auto"
    segment_batch_size: int = DEFAULT_SEGMENT_BATCH_SIZE
    min_segment_ms: int = 300
    vad_max_segment_ms: int = 15000
    vad_max_end_silence_ms: int = 500
    merge_target_segment_ms: int = 4000
    merge_max_segment_ms: int = 12000
    merge_max_gap_ms: int = 250
    prep_workers: int = DEFAULT_PREP_WORKERS
    vad_workers: int = DEFAULT_VAD_WORKERS
    vad_device: str = "auto"
    asr_prefetch_batches: int = DEFAULT_ASR_PREFETCH_BATCHES
    vad_cache_dir: Path = Path(".cache/qwen_srt_vad")
    use_vad_cache: bool = True
