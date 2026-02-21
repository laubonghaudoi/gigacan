from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ASR_ENGINE = "qwen3"
DEFAULT_ASR_MODEL = "iic/SenseVoiceSmall"
DEFAULT_ASR_LANGUAGE = "yue"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_LANGUAGE = "Cantonese"
DEFAULT_QWEN_CONTEXT_PROMPT = (
    "TODO: Replace this placeholder with your Cantonese transcription context prompt."
)
DEFAULT_SEGMENT_BATCH_SIZE = 1536
DEFAULT_SUPER_BATCH_ACTIVE_FILES = 48
DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER = 48
DEFAULT_SUPER_BATCH_PRELOAD_FILES = 96
DEFAULT_SUPER_BATCH_MAX_DECODED_GIB = 25.0
DEFAULT_PREP_WORKERS = 24
DEFAULT_VAD_WORKERS = 24
DEFAULT_ASR_PREFETCH_BATCHES = 6
DEFAULT_VAD_MAX_SEGMENT_MS = 20000
DEFAULT_VAD_MAX_END_SILENCE_MS = 300
DEFAULT_VLLM_MAX_MODEL_LEN = 4096
DEFAULT_VLLM_MAX_NUM_SEQS = 256


@dataclass(slots=True)
class TranscribeConfig:
    """Configuration for one audio-to-SRT transcription run."""

    audio: Path
    output_srt: Path
    asr_engine: str = DEFAULT_ASR_ENGINE
    asr_model: str = DEFAULT_ASR_MODEL
    asr_model_hub: str = "auto"
    asr_language: str = DEFAULT_ASR_LANGUAGE
    asr_use_itn: bool = True
    qwen_src_dir: Path = Path(".cache/Qwen3-ASR-src")
    qwen_repo_url: str = "https://github.com/QwenLM/Qwen3-ASR"
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_language: str = DEFAULT_QWEN_LANGUAGE
    qwen_context: str = DEFAULT_QWEN_CONTEXT_PROMPT
    use_prompt: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    vllm_max_num_seqs: int = DEFAULT_VLLM_MAX_NUM_SEQS
    qwen_max_new_tokens: int = 256
    device: str = "auto"
    segment_batch_size: int = DEFAULT_SEGMENT_BATCH_SIZE
    min_segment_ms: int = 300
    vad_max_segment_ms: int = DEFAULT_VAD_MAX_SEGMENT_MS
    vad_max_end_silence_ms: int = DEFAULT_VAD_MAX_END_SILENCE_MS
    merge_target_segment_ms: int = 4000
    merge_max_segment_ms: int = 12000
    merge_max_gap_ms: int = 250
    prep_workers: int = DEFAULT_PREP_WORKERS
    vad_workers: int = DEFAULT_VAD_WORKERS
    vad_device: str = "auto"
    asr_prefetch_batches: int = DEFAULT_ASR_PREFETCH_BATCHES
    vad_cache_dir: Path = Path(".cache/qwen_srt_vad")
    use_vad_cache: bool = True
