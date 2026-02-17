from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import DEFAULT_QWEN_CONTEXT_PROMPT, TranscribeConfig
from .pipeline import transcribe_to_srt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file with Qwen3-ASR-1.7B and output SRT."
    )
    parser.add_argument("--audio", required=True, help="Input audio path.")
    parser.add_argument(
        "--output-srt",
        default="",
        help="Output SRT path. Default: same stem as input with .srt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto" (default), "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=0,
        help="ASR batch size for VAD segments. 0 = auto (GPU:128, CPU:4).",
    )
    parser.add_argument(
        "--min-segment-ms",
        type=int,
        default=300,
        help="Skip VAD segments shorter than this (ms).",
    )
    parser.add_argument(
        "--vad-max-segment-ms",
        type=int,
        default=20000,
        help="Maximum VAD segment duration in ms. Default: 20000 (20s).",
    )
    parser.add_argument(
        "--qwen-src-dir",
        default=".cache/Qwen3-ASR-src",
        help='Path to Qwen3-ASR source code. Auto-cloned if missing. Default: ".cache/Qwen3-ASR-src".',
    )
    parser.add_argument(
        "--qwen-repo-url",
        default="https://github.com/QwenLM/Qwen3-ASR",
        help="Git URL used when auto-cloning Qwen3-ASR source.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-ASR-1.7B",
        help='Hugging Face model id or local path. Default: "Qwen/Qwen3-ASR-1.7B".',
    )
    parser.add_argument(
        "--qwen-language",
        default="Cantonese",
        help='Forced language for Qwen ASR. Default: "Cantonese".',
    )
    parser.add_argument(
        "--qwen-context",
        default=DEFAULT_QWEN_CONTEXT_PROMPT,
        help="Context prompt passed to Qwen3-ASR for each segment. Edit the placeholder default in this script or override with this argument.",
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Enable context prompt in decoding. Default: disabled (no prompt).",
    )
    parser.add_argument(
        "--qwen-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help='Model dtype. "auto" uses bfloat16 on CUDA and float32 on CPU.',
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per segment batch.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TranscribeConfig:
    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)

    audio_path = Path(namespace.audio)
    output_srt_path = (
        Path(namespace.output_srt)
        if namespace.output_srt
        else audio_path.with_suffix(".srt")
    )

    return TranscribeConfig(
        audio=audio_path,
        output_srt=output_srt_path,
        device=namespace.device,
        segment_batch_size=namespace.segment_batch_size,
        min_segment_ms=namespace.min_segment_ms,
        vad_max_segment_ms=namespace.vad_max_segment_ms,
        qwen_src_dir=Path(namespace.qwen_src_dir),
        qwen_repo_url=namespace.qwen_repo_url,
        qwen_model=namespace.qwen_model,
        qwen_language=namespace.qwen_language,
        qwen_context=namespace.qwen_context,
        use_prompt=namespace.use_prompt,
        qwen_dtype=namespace.qwen_dtype,
        qwen_max_new_tokens=namespace.qwen_max_new_tokens,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    transcribe_to_srt(config)
