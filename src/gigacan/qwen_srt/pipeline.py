from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Thread, local
from typing import Any

import numpy as np

from .audio import (
    compute_vad_streaming,
    load_audio_and_vad_streaming,
    load_audio_mono_16k,
    slice_audio_segment,
)
from .batch import BatchJob, sort_jobs_by_duration
from .config import TranscribeConfig
from .postprocess import CantonesePostProcessor, clean_asr_text
from .runtime import (
    build_asr_model,
    build_vad_model,
    resolve_device,
    resolve_segment_batch_size,
)
from .srt import write_srt
from .vad_cache import load_vad_cache, save_vad_cache


@dataclass(slots=True)
class PreparedTranscriber:
    """Reusable ASR+VAD runtime for transcribing multiple files."""

    asr_model: Any
    vad_model: Any
    postprocessor: CantonesePostProcessor
    device: str
    vad_device: str
    segment_batch_size: int
    min_segment_ms: int
    vad_max_segment_ms: int
    vad_max_end_silence_ms: int
    merge_target_segment_ms: int
    merge_max_segment_ms: int
    merge_max_gap_ms: int
    prep_workers: int
    vad_workers: int
    asr_prefetch_batches: int
    vad_cache_dir: Path
    use_vad_cache: bool
    asr_engine: str
    asr_language: str
    asr_context: str


@dataclass(slots=True)
class FileBatchState:
    job: BatchJob
    segments: list[tuple[int, int]]
    next_segment_idx: int = 0
    entries: list[tuple[int, int, str]] = field(default_factory=list)
    audio_samples: Any | None = None
    audio_sr: int = 0
    decoded_audio_bytes: int = 0
    pending_tasks: int = 0
    failed: bool = False
    done: bool = False


@dataclass(slots=True)
class SegmentTask:
    state: FileBatchState
    start_ms: int
    end_ms: int
    duration_ms: int
    estimated_frames: int


@dataclass(slots=True)
class PreparedJobItem:
    job: BatchJob | None = None
    state: FileBatchState | None = None
    failure: str | None = None
    stop: bool = False


@dataclass(slots=True)
class BatchBuildResult:
    batch_audio: list[tuple[Any, int]]
    batch_meta: list[SegmentTask]
    failures: list[tuple[SegmentTask, str]]
    feats: Any | None = None
    feat_lengths: Any | None = None


@dataclass(slots=True)
class PrefetchedBatch:
    tasks: list[SegmentTask]
    future: Future[BatchBuildResult]


BYTES_PER_GIB = 1024**3
DECODED_AUDIO_BYTES_PER_SECOND = 16_000 * 4  # float32 mono 16kHz


def try_get_total_memory_bytes() -> int:
    """Best-effort total system memory detection."""
    if not hasattr(os, "sysconf"):
        return 0
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages <= 0 or page_size <= 0:
            return 0
        return pages * page_size
    except Exception:
        return 0


def estimate_decoded_audio_bytes(duration_ms: int) -> int:
    safe_duration_ms = max(1, int(duration_ms))
    base = (safe_duration_ms / 1000.0) * DECODED_AUDIO_BYTES_PER_SECOND
    # Add modest headroom for array/object overhead.
    return max(1, int(base * 1.10))


def estimate_loaded_audio_bytes(audio_samples: Any) -> int:
    nbytes = getattr(audio_samples, "nbytes", None)
    if isinstance(nbytes, int) and nbytes > 0:
        return nbytes
    try:
        return max(1, int(len(audio_samples)) * 4)
    except Exception:
        return 1


def estimate_feature_frames(duration_ms: int, frame_ms: int = 10) -> int:
    """Estimate encoder frames from duration in milliseconds."""
    safe_frame_ms = max(1, int(frame_ms))
    safe_duration_ms = max(0, int(duration_ms))
    return max(1, (safe_duration_ms + safe_frame_ms - 1) // safe_frame_ms)


def resolve_context_prompt(config: TranscribeConfig) -> str:
    if config.asr_engine.strip().lower() != "qwen3":
        return ""
    if not config.use_prompt:
        return ""
    prompt = config.qwen_context.strip()
    return prompt


def resolve_vad_device_policy(
    *,
    aux_device: str,
    resolved_vad_workers: int,
    vad_device_policy: str,
) -> str:
    policy = vad_device_policy.strip().lower()
    if policy == "auto":
        return (
            "cpu"
            if aux_device.startswith("cuda") and resolved_vad_workers > 1
            else aux_device
        )
    if policy == "cpu":
        return "cpu"
    if policy == "cuda":
        if not aux_device.startswith("cuda"):
            raise RuntimeError(
                f'--vad-device=cuda requires CUDA ASR device, got "{aux_device}".'
            )
        return aux_device
    raise ValueError(f"Unsupported vad device policy: {vad_device_policy}")


def prepare_transcriber(config: TranscribeConfig) -> PreparedTranscriber:
    resolved_engine = config.asr_engine.strip().lower()
    if resolved_engine not in {"sensevoice", "qwen3"}:
        raise ValueError(f"Unsupported ASR engine: {config.asr_engine}")
    resolved_device = resolve_device(config.device)
    if resolved_engine == "qwen3" and not resolved_device.startswith("cuda"):
        raise RuntimeError(
            f"Qwen3 vLLM requires CUDA, got resolved device {resolved_device!r}. "
            "Use --asr-engine sensevoice on CPU."
        )
    segment_batch_size = resolve_segment_batch_size(
        resolved_device,
        config.segment_batch_size,
    )
    resolved_vad_workers = (
        config.vad_workers
        if config.vad_workers > 0
        else (4 if resolved_device.startswith("cuda") else 2)
    )
    aux_device = (
        "cuda:0"
        if resolved_engine == "qwen3" and resolved_device.startswith("cuda")
        else resolved_device
    )
    vad_device = resolve_vad_device_policy(
        aux_device=aux_device,
        resolved_vad_workers=resolved_vad_workers,
        vad_device_policy=config.vad_device,
    )
    context_prompt = resolve_context_prompt(config)
    asr_language = (
        config.asr_language
        if resolved_engine == "sensevoice"
        else config.qwen_language
    )
    import torch as _torch

    saved_threads = _torch.get_num_threads()
    asr_model = build_asr_model(config, resolved_device, segment_batch_size)

    print(f"ASR engine: {resolved_engine}")
    if resolved_engine == "sensevoice":
        print(f"ASR model: {config.asr_model} ({config.asr_model_hub})")
        print(f"ASR ITN: {'enabled' if config.asr_use_itn else 'disabled'}")
    else:
        print(f"ASR model: {config.qwen_model} (vllm)")
        print(
            "vLLM settings: "
            f"gpu_memory_utilization={config.vllm_gpu_memory_utilization}, "
            f"tensor_parallel_size={config.vllm_tensor_parallel_size}, "
            f"max_model_len={config.vllm_max_model_len}, "
            f"max_num_seqs={config.vllm_max_num_seqs}"
        )
        print(f"Qwen prompt: {'enabled' if bool(context_prompt) else 'disabled'}")
    print(f"Using device: {resolved_device}")
    print(f"Using VAD device: {vad_device}")
    print(f"ASR language: {asr_language}")
    print(f"VAD max end silence: {config.vad_max_end_silence_ms} ms")
    print(f"Segment batch size: {segment_batch_size}")

    vad_model = build_vad_model(
        vad_device,
        config.vad_max_segment_ms,
        config.vad_max_end_silence_ms,
    )

    # FunASR's AutoModel.__init__ calls torch.set_num_threads(4), clobbering
    # the process-global thread count.  Restore it so our fbank pool and other
    # torch CPU ops use all available cores.
    if _torch.get_num_threads() != saved_threads:
        _torch.set_num_threads(saved_threads)
    return PreparedTranscriber(
        asr_model=asr_model,
        vad_model=vad_model,
        postprocessor=CantonesePostProcessor(),
        device=resolved_device,
        vad_device=vad_device,
        segment_batch_size=segment_batch_size,
        min_segment_ms=config.min_segment_ms,
        vad_max_segment_ms=config.vad_max_segment_ms,
        vad_max_end_silence_ms=config.vad_max_end_silence_ms,
        merge_target_segment_ms=config.merge_target_segment_ms,
        merge_max_segment_ms=config.merge_max_segment_ms,
        merge_max_gap_ms=config.merge_max_gap_ms,
        prep_workers=config.prep_workers,
        vad_workers=config.vad_workers,
        asr_prefetch_batches=config.asr_prefetch_batches,
        vad_cache_dir=config.vad_cache_dir,
        use_vad_cache=config.use_vad_cache,
        asr_engine=resolved_engine,
        asr_language=asr_language,
        asr_context=context_prompt,
    )


def collect_vad_segments(
    runtime: PreparedTranscriber,
    audio: Path,
    *,
    vad_model: Any | None = None,
    audio_samples: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    base_segments: list[tuple[int, int]] | None = None
    if runtime.use_vad_cache:
        cached = load_vad_cache(
            runtime.vad_cache_dir,
            audio,
            min_segment_ms=runtime.min_segment_ms,
            vad_max_segment_ms=runtime.vad_max_segment_ms,
            vad_max_end_silence_ms=runtime.vad_max_end_silence_ms,
        )
        if cached is not None:
            base_segments = cached

    if base_segments is None:
        active_vad_model = runtime.vad_model if vad_model is None else vad_model
        vad_input: str | np.ndarray = (
            audio_samples if audio_samples is not None else str(audio)
        )
        vad_res = active_vad_model.generate(input=vad_input)
        if not vad_res or "value" not in vad_res[0]:
            raise RuntimeError(f"Unexpected VAD output for {audio}: {vad_res}")
        raw_segments = vad_res[0]["value"]
        base_segments = [
            (int(start), int(end))
            for start, end in raw_segments
            if int(end) - int(start) >= runtime.min_segment_ms
        ]
        if runtime.use_vad_cache:
            save_vad_cache(
                runtime.vad_cache_dir,
                audio,
                min_segment_ms=runtime.min_segment_ms,
                vad_max_segment_ms=runtime.vad_max_segment_ms,
                vad_max_end_silence_ms=runtime.vad_max_end_silence_ms,
                segments=base_segments,
            )

    return merge_small_vad_segments(
        base_segments,
        target_segment_ms=runtime.merge_target_segment_ms,
        max_segment_ms=runtime.merge_max_segment_ms,
        max_gap_ms=runtime.merge_max_gap_ms,
    )


def merge_small_vad_segments(
    segments: Sequence[tuple[int, int]],
    *,
    target_segment_ms: int,
    max_segment_ms: int,
    max_gap_ms: int,
) -> list[tuple[int, int]]:
    if target_segment_ms <= 0 or len(segments) <= 1:
        return list(segments)

    merged: list[tuple[int, int]] = []
    cur_start, cur_end = int(segments[0][0]), int(segments[0][1])

    for start, end in segments[1:]:
        start_ms = int(start)
        end_ms = int(end)
        if end_ms <= start_ms:
            continue

        gap_ms = start_ms - cur_end
        cur_duration_ms = cur_end - cur_start
        next_duration_ms = end_ms - start_ms
        combined_duration_ms = end_ms - cur_start

        should_merge = (
            gap_ms <= max_gap_ms
            and combined_duration_ms <= max_segment_ms
            and (
                cur_duration_ms < target_segment_ms
                or next_duration_ms < target_segment_ms
            )
        )
        if should_merge:
            cur_end = end_ms
            continue

        merged.append((cur_start, cur_end))
        cur_start, cur_end = start_ms, end_ms

    merged.append((cur_start, cur_end))
    return merged


def resolve_super_batch_active_files(
    runtime: PreparedTranscriber,
    max_active_files: int,
) -> int:
    if max_active_files > 0:
        return max_active_files

    if runtime.device.startswith("cuda"):
        return max(3, min(8, runtime.segment_batch_size // 32))
    return max(2, min(6, runtime.segment_batch_size // 2))


def resolve_logical_cpu_count() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        return 1
    return cpu_count


def resolve_cpu_worker_budget(runtime: PreparedTranscriber) -> int:
    cpu_count = resolve_logical_cpu_count()
    if runtime.device.startswith("cuda"):
        # Leave room for system processes to keep long runs stable.
        reserve_cores = max(2, cpu_count // 4)
    else:
        reserve_cores = 1 if cpu_count >= 4 else 0
    return max(1, cpu_count - reserve_cores)


def resolve_preload_files(max_active_files: int, preload_files: int) -> int:
    if preload_files > 0:
        return max(preload_files, max_active_files)
    # Conservative queue depth to avoid decoding too many long files in memory.
    return max_active_files + 2


def resolve_decoded_audio_budget_bytes(
    runtime: PreparedTranscriber,
    max_decoded_gib: float,
) -> int:
    if max_decoded_gib > 0:
        return max(1, int(max_decoded_gib * BYTES_PER_GIB))

    total_memory = try_get_total_memory_bytes()
    if runtime.device.startswith("cuda"):
        default_budget = 6 * BYTES_PER_GIB
        if total_memory > 0:
            ratio_budget = int(total_memory * 0.22)
            return max(2 * BYTES_PER_GIB, min(12 * BYTES_PER_GIB, ratio_budget))
        return default_budget

    default_budget = 2 * BYTES_PER_GIB
    if total_memory > 0:
        ratio_budget = int(total_memory * 0.30)
        return max(1 * BYTES_PER_GIB, min(8 * BYTES_PER_GIB, ratio_budget))
    return default_budget


def resolve_prep_workers(runtime: PreparedTranscriber) -> int:
    if runtime.prep_workers > 0:
        return runtime.prep_workers
    cpu_budget = resolve_cpu_worker_budget(runtime)
    if runtime.device.startswith("cuda"):
        return max(2, min(6, cpu_budget // 3))
    return 1


def resolve_vad_workers(runtime: PreparedTranscriber) -> int:
    if runtime.vad_workers > 0:
        return runtime.vad_workers
    cpu_budget = resolve_cpu_worker_budget(runtime)
    prep_workers = resolve_prep_workers(runtime)
    if runtime.device.startswith("cuda"):
        # Spend remaining CPU budget on VAD to keep the GPU input queue filled.
        remaining_workers = max(2, cpu_budget - prep_workers)
        return max(2, min(8, remaining_workers))
    return 2


def resolve_prefetch_batches(runtime: PreparedTranscriber) -> int:
    return max(1, runtime.asr_prefetch_batches)


def resolve_segment_queue_capacity(
    segment_batch_size: int,
    queue_multiplier: int,
) -> int:
    if queue_multiplier < 1:
        raise ValueError("queue_multiplier must be >= 1")
    return max(segment_batch_size, segment_batch_size * queue_multiplier)


def resolve_per_file_enqueue(
    *,
    segment_batch_size: int,
    configured_active_files: int,
    current_feed_states: int,
) -> int:
    """Adapt per-file queue contribution to keep ASR batches dense.

    We keep fairness when many files are active, but allow aggressive fill when
    the pipeline tails down to just a few remaining files.
    """
    if segment_batch_size < 1:
        raise ValueError("segment_batch_size must be >= 1")
    if configured_active_files < 1:
        raise ValueError("configured_active_files must be >= 1")
    if current_feed_states < 0:
        raise ValueError("current_feed_states must be >= 0")

    # Conservative baseline: fair share when all files are active.
    baseline = max(1, segment_batch_size // (configured_active_files * 2))
    # Adaptive target: when feed states shrink, raise contribution so batches
    # stay full.  With just 1 active file the entire batch should come from it.
    active_feeders = max(1, current_feed_states)
    adaptive = max(1, segment_batch_size // active_feeders)
    # Cap scales with how many feeders are active: many feeders -> tight cap to
    # keep fairness; few feeders -> allow up to segment_batch_size so the GPU
    # gets full-sized batches.
    cap = max(
        baseline,
        max(64, segment_batch_size // max(1, active_feeders)),
    )
    return min(max(baseline, adaptive), cap)


def resolve_frame_budget(
    segment_queue: list[SegmentTask],
    batch_size: int,
) -> int:
    """Resolve a dynamic frame budget to keep batches dense but stable."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not segment_queue:
        return 0

    sorted_frames = sorted(task.estimated_frames for task in segment_queue)
    max_frame = sorted_frames[-1]

    # Heuristic: use a mid-high quantile frame count as target per item.
    quantile_idx = int((len(sorted_frames) - 1) * 0.65)
    quantile_frame = sorted_frames[quantile_idx]
    per_item_target = max(80, min(1200, quantile_frame))
    target_items = min(batch_size, len(segment_queue))
    return max(max_frame, per_item_target * target_items)


def select_frame_aware_batch(
    segment_queue: list[SegmentTask],
    batch_size: int,
) -> list[SegmentTask]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not segment_queue:
        return []

    ordered_indices = sorted(
        range(len(segment_queue)),
        key=lambda idx: (
            segment_queue[idx].estimated_frames,
            segment_queue[idx].duration_ms,
        ),
    )
    ordered_frames = [segment_queue[idx].estimated_frames for idx in ordered_indices]
    max_total_frames = resolve_frame_budget(segment_queue, batch_size)

    best_window: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    left = 0
    window_frames = 0

    for right, frame_count in enumerate(ordered_frames):
        window_frames += frame_count
        while left <= right and (
            (right - left + 1) > batch_size or window_frames > max_total_frames
        ):
            window_frames -= ordered_frames[left]
            left += 1

        if left > right:
            continue

        count = right - left + 1
        max_frame = ordered_frames[right]
        padding_waste = (max_frame * count) - window_frames
        score = (
            count,           # prioritize bigger batches
            -padding_waste,  # then lower padding waste
            window_frames,   # then better budget utilization
        )
        if best_score is None or score > best_score:
            best_score = score
            best_window = (left, right)

    if best_window is None:
        selected_idx = {ordered_indices[0]}
    else:
        start, end = best_window
        selected_idx = set(ordered_indices[start : end + 1])

    selected: list[SegmentTask] = []
    remaining: list[SegmentTask] = []
    for idx, task in enumerate(segment_queue):
        if idx in selected_idx:
            selected.append(task)
        else:
            remaining.append(task)
    segment_queue[:] = remaining
    selected.sort(key=lambda task: (task.estimated_frames, task.duration_ms))
    return selected


def transcribe_audio_to_srt(
    runtime: PreparedTranscriber,
    audio: Path,
    output_srt: Path,
    *,
    log_segments: bool = True,
) -> Path:
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    segments = collect_vad_segments(runtime, audio)
    if log_segments:
        print(f"VAD segments: {len(segments)}; used: {len(segments)}")

    audio_samples, audio_sr = load_audio_mono_16k(audio)

    entries: list[tuple[int, int, str]] = []
    for batch_start in range(0, len(segments), runtime.segment_batch_size):
        batch_segments = segments[batch_start : batch_start + runtime.segment_batch_size]
        batch_audio = [
            (slice_audio_segment(audio_samples, audio_sr, start_ms, end_ms), audio_sr)
            for start_ms, end_ms in batch_segments
        ]

        results = runtime.asr_model.transcribe(
            audio=batch_audio,
            context=runtime.asr_context,
            language=runtime.asr_language,
        )
        if len(results) != len(batch_segments):
            raise RuntimeError(
                f"ASR result size mismatch: got {len(results)}, expected {len(batch_segments)}"
            )

        for item, (start_ms, end_ms) in zip(results, batch_segments):
            raw_text = str(getattr(item, "text", ""))
            text = clean_asr_text(raw_text)
            text = runtime.postprocessor.apply(text)
            if text:
                entries.append((start_ms, end_ms, text))

        done = min(batch_start + len(batch_segments), len(segments))
        if log_segments:
            print(f"Transcribed {done}/{len(segments)} segments")

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    write_srt(output_srt, entries)
    if log_segments:
        print(f"Wrote SRT: {output_srt}")
    return output_srt


# ---------------------------------------------------------------------------
# Multiprocessing VAD pre-computation
# ---------------------------------------------------------------------------

def _vad_worker_init(
    vad_device: str,
    vad_max_segment_ms: int,
    vad_max_end_silence_ms: int,
) -> None:
    """Initializer for each multiprocessing worker process."""
    import torch as _torch
    _torch.set_num_threads(1)

    global _mp_vad_model  # noqa: PLW0603
    _mp_vad_model = build_vad_model(
        vad_device,
        vad_max_segment_ms,
        vad_max_end_silence_ms,
    )


def _vad_worker_fn(
    args: tuple[str, int, str, int, int],
) -> tuple[str, int, int, bool]:
    """Process one file: streaming decode+VAD, save to cache.

    Returns (path, n_segments, file_bytes, ok).
    """
    audio_path_str, min_segment_ms, cache_dir_str, vad_max_segment_ms, vad_max_end_silence_ms = args
    audio_path = Path(audio_path_str)
    cache_dir = Path(cache_dir_str)
    file_bytes = 0
    try:
        file_bytes = audio_path.stat().st_size
        cached = load_vad_cache(
            cache_dir, audio_path,
            min_segment_ms=min_segment_ms,
            vad_max_segment_ms=vad_max_segment_ms,
            vad_max_end_silence_ms=vad_max_end_silence_ms,
        )
        if cached is not None:
            return (audio_path_str, len(cached), file_bytes, True)

        raw_segments = compute_vad_streaming(audio_path, _mp_vad_model)  # type: ignore[name-defined]  # noqa: F821
        base_segments = [
            (s, e) for s, e in raw_segments if e - s >= min_segment_ms
        ]
        save_vad_cache(
            cache_dir, audio_path,
            min_segment_ms=min_segment_ms,
            vad_max_segment_ms=vad_max_segment_ms,
            vad_max_end_silence_ms=vad_max_end_silence_ms,
            segments=base_segments,
        )
        return (audio_path_str, len(base_segments), file_bytes, True)
    except Exception as exc:
        print(f"VAD failed for {audio_path}: {type(exc).__name__}: {exc}")
        return (audio_path_str, 0, file_bytes, False)


def precompute_vad_multiprocessing(
    jobs: Sequence[BatchJob],
    runtime: PreparedTranscriber,
    *,
    max_workers: int | None = None,
) -> None:
    """Pre-compute VAD for all files using multiprocessing (bypasses GIL).

    Each worker process has its own VAD model and Python GIL, allowing
    true parallelism for the CPU-bound FSMN-VAD computation.  Results
    are saved to the VAD cache so the main pipeline gets instant cache
    hits and only needs to run ffmpeg decode (subprocess, GIL-free).
    """
    import multiprocessing as mp
    import time as _time

    if max_workers is None:
        max_workers = max(1, min(8, resolve_logical_cpu_count()))

    audio_paths = [str(j.audio) for j in jobs if j.audio.is_file()]
    if not audio_paths:
        return

    cache_dir = str(runtime.vad_cache_dir)
    min_seg = runtime.min_segment_ms
    vad_max_seg = runtime.vad_max_segment_ms
    vad_silence = runtime.vad_max_end_silence_ms
    vad_device = runtime.vad_device

    already_cached = 0
    to_compute: list[str] = []
    for p in audio_paths:
        cached = load_vad_cache(
            Path(cache_dir), Path(p),
            min_segment_ms=min_seg,
            vad_max_segment_ms=vad_max_seg,
            vad_max_end_silence_ms=vad_silence,
        )
        if cached is not None:
            already_cached += 1
        else:
            to_compute.append(p)

    to_compute.sort(key=lambda p: Path(p).stat().st_size)

    if not to_compute:
        print(f"VAD pre-computation: all {already_cached} files already cached, skipping.")
        return

    total_bytes = sum(Path(p).stat().st_size for p in to_compute)

    print(
        f"VAD pre-computation: {already_cached} cached, "
        f"{len(to_compute)} to compute ({total_bytes / 1e9:.1f} GB) "
        f"using {max_workers} processes..."
    )
    t0 = _time.monotonic()

    from tqdm import tqdm

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max_workers,
        initializer=_vad_worker_init,
        initargs=(vad_device, vad_max_seg, vad_silence),
    ) as pool:
        task_args = [
            (p, min_seg, cache_dir, vad_max_seg, vad_silence)
            for p in to_compute
        ]
        failed = 0
        bytes_done = 0
        total_segments = 0

        with tqdm(
            total=len(to_compute),
            unit="file",
            desc="VAD pre-compute",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} files "
                "[{elapsed}<{remaining}, {rate_fmt}, "
                "{postfix}]"
            ),
        ) as pbar:
            for path_str, n_segs, file_bytes, ok in pool.imap_unordered(
                _vad_worker_fn, task_args, chunksize=8,
            ):
                if not ok:
                    failed += 1
                bytes_done += file_bytes
                total_segments += n_segs
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{bytes_done / 1e9:.1f}/{total_bytes / 1e9:.0f}GB "
                    f"{total_segments:,}segs "
                    f"{failed}err",
                    refresh=False,
                )

    elapsed = _time.monotonic() - t0
    print(
        f"VAD pre-computation done: {len(to_compute)} files "
        f"({total_bytes / 1e9:.1f} GB) in {elapsed:.1f}s "
        f"({len(to_compute) / elapsed:.1f} files/s), "
        f"{total_segments:,} segments, {failed} failed."
    )


def transcribe_batch_to_srt_superbatched(
    runtime: PreparedTranscriber,
    jobs: Sequence[BatchJob],
    *,
    max_active_files: int,
    queue_multiplier: int = 4,
    preload_files: int = 0,
    max_decoded_gib: float = 0.0,
    continue_on_error: bool,
    on_file_done: Callable[[Path], None] | None = None,
    on_file_failed: Callable[[Path, str], None] | None = None,
) -> tuple[int, list[tuple[Path, str]]]:
    if max_active_files < 0:
        raise ValueError("max_active_files must be >= 0")
    if preload_files < 0:
        raise ValueError("preload_files must be >= 0")
    if max_decoded_gib < 0:
        raise ValueError("max_decoded_gib must be >= 0")

    resolved_active_files = resolve_super_batch_active_files(runtime, max_active_files)
    resolved_preload_files = resolve_preload_files(resolved_active_files, preload_files)
    decoded_budget_bytes = resolve_decoded_audio_budget_bytes(
        runtime,
        max_decoded_gib,
    )
    resolved_prep_workers = resolve_prep_workers(runtime)
    resolved_vad_workers = resolve_vad_workers(runtime)
    vad_worker_device = runtime.vad_device
    resolved_prefetch_batches = resolve_prefetch_batches(runtime)
    segment_queue_capacity = resolve_segment_queue_capacity(
        runtime.segment_batch_size,
        queue_multiplier,
    )
    default_per_file_enqueue = resolve_per_file_enqueue(
        segment_batch_size=runtime.segment_batch_size,
        configured_active_files=resolved_active_files,
        current_feed_states=resolved_active_files,
    )

    print(
        "Super-batch settings: "
        f"cpu_count={resolve_logical_cpu_count()}, "
        f"active_files={resolved_active_files}, "
        f"preload_files={resolved_preload_files}, "
        f"prep_workers={resolved_prep_workers}, "
        f"vad_workers={resolved_vad_workers}, "
        f"vad_device={vad_worker_device}, "
        f"decoded_budget_gib={decoded_budget_bytes / BYTES_PER_GIB:.2f}, "
        f"prefetch_batches={resolved_prefetch_batches}, "
        f"segment_queue={segment_queue_capacity}, "
        f"per_file_enqueue_default={default_per_file_enqueue}"
    )

    ordered_jobs = sort_jobs_by_duration(jobs)

    if runtime.use_vad_cache:
        precompute_vad_multiprocessing(ordered_jobs, runtime)

    completed = 0
    failures: list[tuple[Path, str]] = []
    prepared_queue: Queue[PreparedJobItem] = Queue(maxsize=resolved_preload_files)
    producer_done = False
    feed_states: deque[FileBatchState] = deque()
    live_states: list[FileBatchState] = []
    segment_queue: list[SegmentTask] = []
    decoded_budget_cv = Condition()
    decoded_bytes_accounted = 0
    use_direct_inference: bool = getattr(runtime.asr_model, "_direct_ready", False)

    def try_reserve_decoded_budget(requested_bytes: int) -> int | None:
        """Try to reserve budget without blocking. Returns reserved bytes or None."""
        nonlocal decoded_bytes_accounted
        reserved = max(1, int(requested_bytes))
        with decoded_budget_cv:
            if (
                decoded_bytes_accounted + reserved <= decoded_budget_bytes
                or decoded_bytes_accounted == 0
            ):
                decoded_bytes_accounted += reserved
                return reserved
        return None

    def release_decoded_budget(released_bytes: int) -> None:
        nonlocal decoded_bytes_accounted
        released = max(0, int(released_bytes))
        if released == 0:
            return
        with decoded_budget_cv:
            decoded_bytes_accounted = max(0, decoded_bytes_accounted - released)
            decoded_budget_cv.notify_all()

    def adjust_decoded_budget(delta_bytes: int) -> None:
        nonlocal decoded_bytes_accounted
        delta = int(delta_bytes)
        if delta == 0:
            return
        with decoded_budget_cv:
            decoded_bytes_accounted = max(0, decoded_bytes_accounted + delta)
            if delta < 0:
                decoded_budget_cv.notify_all()

    def release_state_audio(state: FileBatchState) -> None:
        released = int(state.decoded_audio_bytes)
        state.audio_samples = None
        state.decoded_audio_bytes = 0
        release_decoded_budget(released)

    def handle_failure(state: FileBatchState, message: str) -> None:
        if state.failed or state.done:
            return
        state.failed = True
        state.entries.clear()
        release_state_audio(state)
        failures.append((state.job.audio, message))
        if on_file_failed is not None:
            on_file_failed(state.job.audio, message)

    def handle_job_failure(job: BatchJob, message: str) -> None:
        failures.append((job.audio, message))
        if on_file_failed is not None:
            on_file_failed(job.audio, message)

    def finalize_state(state: FileBatchState) -> None:
        nonlocal completed
        if state.failed or state.done:
            return
        state.job.output_srt.parent.mkdir(parents=True, exist_ok=True)
        state.entries.sort()
        write_srt(state.job.output_srt, state.entries)
        release_state_audio(state)
        state.done = True
        completed += 1
        if on_file_done is not None:
            on_file_done(state.job.audio)

    def write_empty_output(job: BatchJob) -> None:
        nonlocal completed
        job.output_srt.parent.mkdir(parents=True, exist_ok=True)
        write_srt(job.output_srt, [])
        completed += 1
        if on_file_done is not None:
            on_file_done(job.audio)

    def producer() -> None:
        PrepResult = tuple[list[tuple[int, int]], np.ndarray, int]
        pending_jobs: dict[Future[PrepResult], tuple[BatchJob, int]] = {}
        backlog_limit = min(
            resolved_preload_files,
            max(
                resolved_active_files + resolved_prep_workers,
                resolved_prep_workers * 2,
            ),
        )
        vad_local = local()

        def estimate_file_decode_bytes(audio_path: Path) -> int:
            """Estimate decoded 16kHz mono float32 size from file size."""
            file_bytes = audio_path.stat().st_size
            return max(1, int(file_bytes * 5.5))

        def _ensure_worker_model() -> Any:
            import torch as _torch

            model = getattr(vad_local, "model", None)
            if model is None:
                model = build_vad_model(
                    vad_worker_device,
                    runtime.vad_max_segment_ms,
                    runtime.vad_max_end_silence_ms,
                )
                setattr(vad_local, "model", model)
                _torch.set_num_threads(1)
            return model

        def decode_and_vad_for_job(
            job: BatchJob,
        ) -> PrepResult:
            """Decode audio and run VAD, streaming chunks for overlap."""
            if runtime.use_vad_cache:
                cached = load_vad_cache(
                    runtime.vad_cache_dir,
                    job.audio,
                    min_segment_ms=runtime.min_segment_ms,
                    vad_max_segment_ms=runtime.vad_max_segment_ms,
                    vad_max_end_silence_ms=runtime.vad_max_end_silence_ms,
                )
                if cached is not None:
                    audio_samples, audio_sr = load_audio_mono_16k(job.audio)
                    segments = merge_small_vad_segments(
                        cached,
                        target_segment_ms=runtime.merge_target_segment_ms,
                        max_segment_ms=runtime.merge_max_segment_ms,
                        max_gap_ms=runtime.merge_max_gap_ms,
                    )
                    return segments, audio_samples, audio_sr

            if resolved_vad_workers <= 1:
                worker_model = runtime.vad_model
            else:
                worker_model = _ensure_worker_model()

            raw_segments, audio_samples, audio_sr = load_audio_and_vad_streaming(
                job.audio, worker_model,
            )

            base_segments = [
                (s, e) for s, e in raw_segments
                if e - s >= runtime.min_segment_ms
            ]

            if runtime.use_vad_cache:
                save_vad_cache(
                    runtime.vad_cache_dir,
                    job.audio,
                    min_segment_ms=runtime.min_segment_ms,
                    vad_max_segment_ms=runtime.vad_max_segment_ms,
                    vad_max_end_silence_ms=runtime.vad_max_end_silence_ms,
                    segments=base_segments,
                )

            segments = merge_small_vad_segments(
                base_segments,
                target_segment_ms=runtime.merge_target_segment_ms,
                max_segment_ms=runtime.merge_max_segment_ms,
                max_gap_ms=runtime.merge_max_gap_ms,
            )
            return segments, audio_samples, audio_sr

        def drain_jobs(*, block: bool) -> None:
            if not pending_jobs:
                return
            done, _ = wait(
                set(pending_jobs),
                timeout=None if block else 0.0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                return
            for future in done:
                job, reserved_bytes = pending_jobs.pop(future)
                try:
                    segments, audio_samples, audio_sr = future.result()
                except Exception as exc:  # noqa: BLE001
                    release_decoded_budget(reserved_bytes)
                    prepared_queue.put(
                        PreparedJobItem(
                            job=job,
                            failure=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue

                if not segments:
                    release_decoded_budget(reserved_bytes)
                    prepared_queue.put(PreparedJobItem(job=job))
                    continue

                actual_bytes = estimate_loaded_audio_bytes(audio_samples)
                if actual_bytes > reserved_bytes:
                    adjust_decoded_budget(actual_bytes - reserved_bytes)
                elif actual_bytes < reserved_bytes:
                    release_decoded_budget(reserved_bytes - actual_bytes)
                prepared_queue.put(
                    PreparedJobItem(
                        state=FileBatchState(
                            job=job,
                            segments=segments,
                            audio_samples=audio_samples,
                            audio_sr=audio_sr,
                            decoded_audio_bytes=actual_bytes,
                        )
                    )
                )

        pool_size = max(resolved_prep_workers, resolved_vad_workers)
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            for job in ordered_jobs:
                if not job.audio.is_file():
                    prepared_queue.put(
                        PreparedJobItem(
                            job=job,
                            failure=f"FileNotFoundError: Audio file not found: {job.audio}",
                        )
                    )
                    continue

                estimated = estimate_file_decode_bytes(job.audio)
                reserved_bytes = try_reserve_decoded_budget(estimated)
                while reserved_bytes is None:
                    drain_jobs(block=True)
                    reserved_bytes = try_reserve_decoded_budget(estimated)

                try:
                    future = pool.submit(decode_and_vad_for_job, job)
                except Exception as exc:  # noqa: BLE001
                    release_decoded_budget(reserved_bytes)
                    prepared_queue.put(
                        PreparedJobItem(
                            job=job,
                            failure=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                pending_jobs[future] = (job, reserved_bytes)

                drain_jobs(block=False)
                if len(pending_jobs) >= backlog_limit:
                    drain_jobs(block=True)

            while pending_jobs:
                drain_jobs(block=True)
        prepared_queue.put(PreparedJobItem(stop=True))

    producer_thread = Thread(target=producer, daemon=True)
    producer_thread.start()

    def consume_prepared(block: bool) -> None:
        nonlocal producer_done
        timeout = 0.2 if block else 0.0
        while len(feed_states) < resolved_active_files and not producer_done:
            try:
                item = prepared_queue.get(timeout=timeout)
            except Empty:
                return

            if item.stop:
                producer_done = True
                return

            if item.failure is not None and item.job is not None:
                if continue_on_error:
                    handle_job_failure(item.job, item.failure)
                else:
                    raise RuntimeError(item.failure)
                continue

            if item.state is not None:
                live_states.append(item.state)
                feed_states.append(item.state)
                continue

            if item.job is not None:
                write_empty_output(item.job)

    def prune_failed_tasks() -> None:
        if not segment_queue:
            return
        kept: list[SegmentTask] = []
        for task in segment_queue:
            if task.state.failed:
                task.state.pending_tasks = max(0, task.state.pending_tasks - 1)
            else:
                kept.append(task)
        segment_queue[:] = kept

    def refill_segment_queue() -> None:
        if not feed_states or len(segment_queue) >= segment_queue_capacity:
            return

        # Target enough queued segments for the full prefetch depth so the GPU
        # never stalls waiting for the next batch to be built.
        target = min(
            segment_queue_capacity,
            runtime.segment_batch_size * resolved_prefetch_batches,
        )
        while feed_states and len(segment_queue) < target:
            state = feed_states.popleft()
            if state.failed or state.done:
                continue

            if state.next_segment_idx >= len(state.segments):
                continue

            remaining = len(state.segments) - state.next_segment_idx
            per_file_enqueue = resolve_per_file_enqueue(
                segment_batch_size=runtime.segment_batch_size,
                configured_active_files=resolved_active_files,
                current_feed_states=len(feed_states) + 1,
            )
            budget = min(
                per_file_enqueue,
                remaining,
                segment_queue_capacity - len(segment_queue),
            )
            for _ in range(budget):
                start_ms, end_ms = state.segments[state.next_segment_idx]
                state.next_segment_idx += 1
                state.pending_tasks += 1
                segment_queue.append(
                    SegmentTask(
                        state=state,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        duration_ms=end_ms - start_ms,
                        estimated_frames=estimate_feature_frames(end_ms - start_ms),
                    )
                )
                if len(segment_queue) >= segment_queue_capacity:
                    break

            if state.next_segment_idx < len(state.segments):
                feed_states.append(state)

    def maybe_finalize_states() -> None:
        for state in list(live_states):
            if state.failed:
                live_states.remove(state)
                continue
            if state.next_segment_idx >= len(state.segments) and state.pending_tasks == 0:
                finalize_state(state)
                live_states.remove(state)

    def build_batch_payload(batch_tasks: list[SegmentTask]) -> BatchBuildResult:
        batch_audio: list[tuple[Any, int]] = []
        batch_meta: list[SegmentTask] = []
        build_failures: list[tuple[SegmentTask, str]] = []

        for task in batch_tasks:
            state = task.state
            if state.failed:
                continue
            try:
                segment_audio = slice_audio_segment(
                    state.audio_samples,
                    state.audio_sr,
                    task.start_ms,
                    task.end_ms,
                )
            except Exception as exc:  # noqa: BLE001
                build_failures.append((task, f"{type(exc).__name__}: {exc}"))
                continue

            batch_audio.append((segment_audio, state.audio_sr))
            batch_meta.append(task)

        # Pre-compute fbank features so that GPU inference receives
        # ready-to-go tensors instead of doing CPU extraction inline.
        feats = None
        feat_lengths = None
        if batch_audio and use_direct_inference:
            try:
                waveforms = [samples for samples, _sr in batch_audio]
                feats, feat_lengths = runtime.asr_model.extract_features(waveforms)
            except Exception:  # noqa: BLE001
                feats = None
                feat_lengths = None

        return BatchBuildResult(
            batch_audio=batch_audio,
            batch_meta=batch_meta,
            failures=build_failures,
            feats=feats,
            feat_lengths=feat_lengths,
        )

    def execute_asr_batch(
        batch_tasks: list[SegmentTask],
        build_result: BatchBuildResult,
    ) -> None:
        batch_audio = build_result.batch_audio
        batch_meta = build_result.batch_meta

        def append_result_text(result_item: Any, task: SegmentTask) -> None:
            state = task.state
            if state.failed:
                return
            raw_text = str(getattr(result_item, "text", ""))
            text = clean_asr_text(raw_text)
            text = runtime.postprocessor.apply(text)
            if text:
                state.entries.append((task.start_ms, task.end_ms, text))

        def transcribe_with_binary_split_fallback(
            items: list[tuple[tuple[Any, int], SegmentTask]],
            root_error: str,
        ) -> None:
            if not items:
                return

            subset_audio = [payload for payload, _ in items]
            subset_meta = [task for _, task in items]
            try:
                subset_results = runtime.asr_model.transcribe(
                    audio=subset_audio,
                    context=runtime.asr_context,
                    language=runtime.asr_language,
                )
                if len(subset_results) != len(subset_meta):
                    raise RuntimeError(
                        "ASR result size mismatch in fallback: "
                        f"got {len(subset_results)}, expected {len(subset_meta)}"
                    )

                for item, task in zip(subset_results, subset_meta):
                    append_result_text(item, task)
                return
            except Exception as subset_exc:  # noqa: BLE001
                if len(items) == 1:
                    failed_task = items[0][1]
                    message = (
                        f"{type(subset_exc).__name__}: {subset_exc} "
                        f"(batch_error={root_error})"
                    )
                    handle_failure(failed_task.state, message)
                    return

            split_at = len(items) // 2
            transcribe_with_binary_split_fallback(items[:split_at], root_error)
            transcribe_with_binary_split_fallback(items[split_at:], root_error)

        try:
            for task, message in build_result.failures:
                if continue_on_error:
                    handle_failure(task.state, message)
                    continue
                raise RuntimeError(message)

            if not batch_audio:
                return

            if build_result.feats is not None:
                batch_results = runtime.asr_model.transcribe_preprocessed(
                    build_result.feats,
                    build_result.feat_lengths,
                    language=runtime.asr_language,
                )
            else:
                batch_results = runtime.asr_model.transcribe(
                    audio=batch_audio,
                    context=runtime.asr_context,
                    language=runtime.asr_language,
                )
            if len(batch_results) != len(batch_meta):
                raise RuntimeError(
                    "ASR result size mismatch: "
                    f"got {len(batch_results)}, expected {len(batch_meta)}"
                )

            for item, task in zip(batch_results, batch_meta):
                append_result_text(item, task)

        except Exception as batch_exc:  # noqa: BLE001
            if not continue_on_error:
                raise

            root_error = f"{type(batch_exc).__name__}: {batch_exc}"
            fallback_items = [
                (payload, task)
                for payload, task in zip(batch_audio, batch_meta)
                if not task.state.failed
            ]
            transcribe_with_binary_split_fallback(fallback_items, root_error)
        finally:
            for task in batch_tasks:
                task.state.pending_tasks = max(0, task.state.pending_tasks - 1)

    build_executor: ThreadPoolExecutor | None = None
    if resolved_prefetch_batches > 1:
        build_executor = ThreadPoolExecutor(max_workers=resolved_prefetch_batches - 1)
    prefetched_batches: deque[PrefetchedBatch] = deque()

    def schedule_prefetch() -> None:
        while len(prefetched_batches) < resolved_prefetch_batches and segment_queue:
            selected = select_frame_aware_batch(
                segment_queue,
                runtime.segment_batch_size,
            )
            if not selected:
                break
            if build_executor is None:
                future: Future[BatchBuildResult] = Future()
                future.set_result(build_batch_payload(selected))
            else:
                future = build_executor.submit(build_batch_payload, selected)
            prefetched_batches.append(PrefetchedBatch(tasks=selected, future=future))

    try:
        while True:
            consume_prepared(block=False)
            prune_failed_tasks()
            refill_segment_queue()
            maybe_finalize_states()
            schedule_prefetch()

            if prefetched_batches:
                prefetched = prefetched_batches.popleft()
                build_result = prefetched.future.result()

                # Schedule the next batch before executing current ASR, so
                # payload construction can overlap with GPU inference.
                consume_prepared(block=False)
                prune_failed_tasks()
                refill_segment_queue()
                maybe_finalize_states()
                schedule_prefetch()

                execute_asr_batch(prefetched.tasks, build_result)
                maybe_finalize_states()
                continue

            if producer_done and not live_states and not feed_states and not segment_queue:
                break

            if not producer_done:
                consume_prepared(block=True)
                continue

            maybe_finalize_states()
            if not live_states and not feed_states and not segment_queue:
                break
    finally:
        if build_executor is not None:
            build_executor.shutdown(wait=False, cancel_futures=True)
        producer_thread.join(timeout=0.1)

    return completed, failures


def transcribe_to_srt(config: TranscribeConfig) -> Path:
    runtime = prepare_transcriber(config)
    return transcribe_audio_to_srt(runtime, config.audio, config.output_srt)
