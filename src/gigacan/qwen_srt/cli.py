from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from .batch import build_batch_jobs, discover_audio_files, filter_audio_files_by_year
from .config import DEFAULT_QWEN_CONTEXT_PROMPT, TranscribeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one audio file or all audio files under a directory "
            "with Qwen3-ASR-1.7B and output SRT."
        )
    )
    parser.add_argument(
        "--audio",
        default="",
        help=(
            "Input audio path (single-file mode). If omitted, batch mode scans "
            "--audio-dir recursively."
        ),
    )
    parser.add_argument(
        "--output-srt",
        default="",
        help="Output SRT path for single-file mode. Default: same stem as --audio with .srt.",
    )
    parser.add_argument(
        "--audio-dir",
        default="download",
        help='Batch input root directory. Default: "download".',
    )
    parser.add_argument(
        "--output-dir",
        default="transcriptions",
        help='Batch output root directory. Default: "transcriptions".',
    )
    parser.add_argument(
        "--year",
        default="",
        help='Batch mode: transcribe only files under download/<year>/ (e.g., "2025").',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SRT files in batch mode. Default: skip existing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch mode when a file fails, then report failures at the end.",
    )
    parser.add_argument(
        "--super-batch-active-files",
        type=int,
        default=0,
        help=(
            "Number of files to keep active for cross-file segment super-batching "
            "in batch mode. 0 = auto-tune (default)."
        ),
    )
    parser.add_argument(
        "--super-batch-queue-multiplier",
        type=int,
        default=4,
        help=(
            "Multiplier for super-batch segment queue capacity relative to "
            "--segment-batch-size. Default: 4."
        ),
    )
    parser.add_argument(
        "--super-batch-preload-files",
        type=int,
        default=0,
        help=(
            "How many files to pre-load (VAD + decoded audio) ahead of GPU "
            "processing in batch mode. 0 = auto-tune (default)."
        ),
    )
    parser.add_argument(
        "--super-batch-max-decoded-gib",
        type=float,
        default=0.0,
        help=(
            "Cap decoded-audio host RAM used by super-batching (GiB). "
            "0 = auto-tune (default)."
        ),
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
        help=(
            "ASR batch size for VAD segments. "
            "0 = auto (GPU:128, CPU:4)."
        ),
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
        "--merge-target-segment-ms",
        type=int,
        default=4000,
        help=(
            "Merge adjacent small VAD segments up to this target duration (ms). "
            "0 disables merging. Default: 4000."
        ),
    )
    parser.add_argument(
        "--merge-max-segment-ms",
        type=int,
        default=12000,
        help=(
            "Upper bound for merged segment duration (ms). "
            "Default: 12000."
        ),
    )
    parser.add_argument(
        "--merge-max-gap-ms",
        type=int,
        default=250,
        help=(
            "Maximum silent gap (ms) allowed when merging adjacent segments. "
            "Default: 250."
        ),
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=0,
        help=(
            "CPU workers for audio decoding prep in batch mode. "
            "0 = auto-tune (default)."
        ),
    )
    parser.add_argument(
        "--vad-workers",
        type=int,
        default=0,
        help=(
            "Workers for VAD generation in batch mode. "
            "0 = auto-tune (default)."
        ),
    )
    parser.add_argument(
        "--asr-prefetch-batches",
        type=int,
        default=2,
        help=(
            "Number of ASR batches to pre-build ahead of inference "
            "for async request feeding. Must be >=1. Default: 2."
        ),
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
        help=(
            'Model dtype for transformers backend. '
            '"auto" uses bfloat16 on CUDA and float32 on CPU.'
        ),
    )
    parser.add_argument(
        "--vad-cache-dir",
        default=".cache/qwen_srt_vad",
        help='Directory for VAD segment cache files. Default: ".cache/qwen_srt_vad".',
    )
    parser.add_argument(
        "--no-vad-cache",
        action="store_true",
        help="Disable VAD cache reads/writes.",
    )
    parser.add_argument(
        "--persistent-worker",
        action="store_true",
        help=(
            "Run batch transcription through a persistent local worker process "
            "to reuse loaded ASR runtime across runs."
        ),
    )
    parser.add_argument(
        "--worker-socket",
        default=".cache/qwen_srt_worker.sock",
        help='UNIX socket path for persistent worker mode. Default: ".cache/qwen_srt_worker.sock".',
    )
    parser.add_argument(
        "--shutdown-worker",
        action="store_true",
        help="Send shutdown command to the persistent worker and exit.",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per segment batch.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    if namespace.audio and namespace.year:
        parser.error("--year is only supported in batch mode (without --audio).")
    if namespace.audio and namespace.persistent_worker:
        parser.error("--persistent-worker is only supported in batch mode.")
    if namespace.super_batch_active_files < 0:
        parser.error("--super-batch-active-files must be >= 0.")
    if namespace.super_batch_queue_multiplier < 1:
        parser.error("--super-batch-queue-multiplier must be >= 1.")
    if namespace.super_batch_preload_files < 0:
        parser.error("--super-batch-preload-files must be >= 0.")
    if namespace.super_batch_max_decoded_gib < 0:
        parser.error("--super-batch-max-decoded-gib must be >= 0.")
    if namespace.merge_target_segment_ms < 0:
        parser.error("--merge-target-segment-ms must be >= 0.")
    if namespace.merge_max_segment_ms < 1:
        parser.error("--merge-max-segment-ms must be >= 1.")
    if namespace.merge_max_gap_ms < 0:
        parser.error("--merge-max-gap-ms must be >= 0.")
    if (
        namespace.merge_target_segment_ms > 0
        and namespace.merge_max_segment_ms < namespace.merge_target_segment_ms
    ):
        parser.error(
            "--merge-max-segment-ms must be >= --merge-target-segment-ms when merging is enabled."
        )
    if namespace.prep_workers < 0:
        parser.error("--prep-workers must be >= 0.")
    if namespace.vad_workers < 0:
        parser.error("--vad-workers must be >= 0.")
    if namespace.asr_prefetch_batches < 1:
        parser.error("--asr-prefetch-batches must be >= 1.")
    return namespace


def build_config(
    namespace: argparse.Namespace,
    *,
    audio: Path,
    output_srt: Path,
) -> TranscribeConfig:
    return TranscribeConfig(
        audio=audio,
        output_srt=output_srt,
        device=namespace.device,
        segment_batch_size=namespace.segment_batch_size,
        min_segment_ms=namespace.min_segment_ms,
        vad_max_segment_ms=namespace.vad_max_segment_ms,
        merge_target_segment_ms=namespace.merge_target_segment_ms,
        merge_max_segment_ms=namespace.merge_max_segment_ms,
        merge_max_gap_ms=namespace.merge_max_gap_ms,
        prep_workers=namespace.prep_workers,
        vad_workers=namespace.vad_workers,
        asr_prefetch_batches=namespace.asr_prefetch_batches,
        qwen_src_dir=Path(namespace.qwen_src_dir),
        qwen_repo_url=namespace.qwen_repo_url,
        qwen_model=namespace.qwen_model,
        qwen_language=namespace.qwen_language,
        qwen_context=namespace.qwen_context,
        use_prompt=namespace.use_prompt,
        qwen_dtype=namespace.qwen_dtype,
        qwen_max_new_tokens=namespace.qwen_max_new_tokens,
        vad_cache_dir=Path(namespace.vad_cache_dir),
        use_vad_cache=not namespace.no_vad_cache,
    )


def run_single_file(namespace: argparse.Namespace) -> None:
    from .pipeline import transcribe_to_srt

    audio_path = Path(namespace.audio)
    output_srt_path = (
        Path(namespace.output_srt)
        if namespace.output_srt
        else audio_path.with_suffix(".srt")
    )
    config = build_config(namespace, audio=audio_path, output_srt=output_srt_path)
    transcribe_to_srt(config)


def run_batch(namespace: argparse.Namespace) -> None:
    from .pipeline import prepare_transcriber, transcribe_batch_to_srt_superbatched
    from .worker_client import transcribe_batch_via_worker

    audio_dir = Path(namespace.audio_dir)
    output_dir = Path(namespace.output_dir)

    all_audio_files = discover_audio_files(audio_dir)
    if namespace.year:
        audio_files = filter_audio_files_by_year(
            all_audio_files,
            audio_dir,
            str(namespace.year),
        )
    else:
        audio_files = all_audio_files

    if not audio_files:
        if namespace.year:
            raise FileNotFoundError(
                f'No audio files found under {audio_dir}/{namespace.year}. '
                "Expected files like .opus, .wav, .mp3, .m4a, .flac, .ogg, .aac, .webm."
            )
        raise FileNotFoundError(
            f"No audio files found under {audio_dir}. "
            "Expected files like .opus, .wav, .mp3, .m4a, .flac, .ogg, .aac, .webm."
        )

    jobs, skipped = build_batch_jobs(
        audio_files,
        audio_dir,
        output_dir,
        overwrite=namespace.overwrite,
    )

    if namespace.year:
        print(
            f"Discovered {len(audio_files)} audio files under {audio_dir}/{namespace.year}"
        )
    else:
        print(f"Discovered {len(audio_files)} audio files under {audio_dir}")
    if skipped:
        print(
            f"Skipping {skipped} files that already have SRT outputs "
            "(use --overwrite to regenerate)."
        )
    if not jobs:
        print("No remaining files to transcribe.")
        return

    first_job = jobs[0]
    config = build_config(
        namespace,
        audio=first_job.audio,
        output_srt=first_job.output_srt,
    )
    runtime = None if namespace.persistent_worker else prepare_transcriber(config)

    completed = 0
    failures: list[tuple[Path, str]] = []

    def to_relative(audio_path: Path) -> str:
        try:
            return str(audio_path.relative_to(audio_dir))
        except ValueError:
            return str(audio_path)

    with tqdm(
        total=len(audio_files),
        initial=skipped,
        unit="file",
        desc="Transcribing",
        dynamic_ncols=True,
    ) as progress:
        def on_done(audio_path: Path) -> None:
            progress.set_postfix_str(to_relative(audio_path))
            progress.update(1)

        def on_failed(audio_path: Path, message: str) -> None:
            progress.set_postfix_str(to_relative(audio_path))
            progress.write(f"Failed: {audio_path} ({message})")
            progress.update(1)

        if namespace.persistent_worker:
            completed, failures = transcribe_batch_via_worker(
                config=config,
                jobs=jobs,
                socket_path=Path(namespace.worker_socket),
                max_active_files=namespace.super_batch_active_files,
                queue_multiplier=namespace.super_batch_queue_multiplier,
                preload_files=namespace.super_batch_preload_files,
                max_decoded_gib=namespace.super_batch_max_decoded_gib,
                continue_on_error=namespace.continue_on_error,
                on_file_done=on_done,
                on_file_failed=on_failed,
            )
        else:
            if runtime is None:
                raise RuntimeError("Runtime initialization failed.")
            completed, failures = transcribe_batch_to_srt_superbatched(
                runtime,
                jobs,
                max_active_files=namespace.super_batch_active_files,
                queue_multiplier=namespace.super_batch_queue_multiplier,
                preload_files=namespace.super_batch_preload_files,
                max_decoded_gib=namespace.super_batch_max_decoded_gib,
                continue_on_error=namespace.continue_on_error,
                on_file_done=on_done,
                on_file_failed=on_failed,
            )

    print(
        f"Batch completed: {completed} transcribed, {skipped} skipped, "
        f"{len(failures)} failed."
    )
    if failures:
        print("Failed files:")
        for audio_path, message in failures[:20]:
            print(f"  - {audio_path}: {message}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise RuntimeError("Batch transcription completed with failures.")


def main(argv: Sequence[str] | None = None) -> None:
    from .worker_client import shutdown_worker

    namespace = parse_args(argv)
    if namespace.shutdown_worker:
        socket_path = Path(namespace.worker_socket)
        stopped = shutdown_worker(socket_path)
        if stopped:
            print(f"Stopped worker at {socket_path}")
        else:
            print(f"No running worker at {socket_path}")
        return
    if namespace.audio:
        run_single_file(namespace)
        return
    run_batch(namespace)
