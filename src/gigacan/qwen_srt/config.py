from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_QWEN_CONTEXT_PROMPT = (
    "TODO: Replace this placeholder with your Cantonese transcription context prompt."
)


@dataclass(slots=True)
class TranscribeConfig:
    """Configuration for one audio-to-SRT transcription run."""

    audio: Path
    output_srt: Path
    device: str = "auto"
    segment_batch_size: int = 0
    min_segment_ms: int = 300
    vad_max_segment_ms: int = 20000
    qwen_src_dir: Path = Path(".cache/Qwen3-ASR-src")
    qwen_repo_url: str = "https://github.com/QwenLM/Qwen3-ASR"
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_language: str = "Cantonese"
    qwen_context: str = DEFAULT_QWEN_CONTEXT_PROMPT
    use_prompt: bool = False
    qwen_dtype: str = "auto"
    qwen_max_new_tokens: int = 256
