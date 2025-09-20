"""Gigacan utilities for segmentation and transcription."""

from .corrector import RuleBasedCorrector  # re-export

__all__ = [
    # segmenter
    "check_ffmpeg",
    "extract_mono_wav",
    "ffprobe_duration_seconds",
    "try_silero_vad_segments",
    "fixed_window_segments",
    "cut_wav_segment",
    "prepare_segments",
    # transcriber
    "RateLimiter",
    "require_dashscope",
    "load_api_key_from_env_or_file",
    "transcribe_segment",
    "transcribe_segments",
    "write_webvtt",
    # corrector
    "RuleBasedCorrector",
]
