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

from .audio import load_audio_mono_16k, slice_audio_segment
from .batch import BatchJob, sort_jobs_by_duration
from .config import TranscribeConfig
from .postprocess import CantonesePostProcessor, clean_asr_text
from .runtime import (
    build_asr_model,
    build_vad_model,
    resolve_backend,
    resolve_device,
    resolve_qwen_dtype,
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
    asr_backend: str
    device: str
    vad_device: str
    segment_batch_size: int
    min_segment_ms: int
    vad_max_segment_ms: int
    merge_target_segment_ms: int
    merge_max_segment_ms: int
    merge_max_gap_ms: int
    prep_workers: int
    vad_workers: int
    asr_prefetch_batches: int
    vad_cache_dir: Path
    use_vad_cache: bool
    qwen_language: str
    context_prompt: str
    use_prompt: bool


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


def resolve_context_prompt(config: TranscribeConfig) -> tuple[str, bool]:
    use_prompt = config.use_prompt and bool(config.qwen_context.strip())
    context_prompt = config.qwen_context if use_prompt else ""
    return context_prompt, use_prompt


def prepare_transcriber(config: TranscribeConfig) -> PreparedTranscriber:
    backend = resolve_backend(config.backend)
    resolved_device = resolve_device(config.device)
    if backend == "transformers":
        qwen_dtype = resolve_qwen_dtype(config.qwen_dtype, resolved_device)
    else:
        qwen_dtype = config.qwen_dtype
    segment_batch_size = resolve_segment_batch_size(
        resolved_device,
        config.segment_batch_size,
        backend=backend,
    )
    resolved_vad_workers = (
        config.vad_workers
        if config.vad_workers > 0
        else (4 if resolved_device.startswith("cuda") else 2)
    )
    vad_device = (
        "cpu"
        if resolved_device.startswith("cuda") and resolved_vad_workers > 1
        else resolved_device
    )
    context_prompt, use_prompt = resolve_context_prompt(config)

    asr_model = build_asr_model(config, resolved_device, qwen_dtype, segment_batch_size)
    actual_backend = str(getattr(asr_model, "_gigacan_backend", backend))

    print(f"ASR backend: {actual_backend}")
    print(f"Using device: {resolved_device}")
    print(f"Using VAD device: {vad_device}")
    print(f"Qwen dtype: {qwen_dtype}")
    print(f"Segment batch size: {segment_batch_size}")
    print(f"Qwen context prompt: {'enabled' if use_prompt else 'disabled'}")

    vad_model = build_vad_model(vad_device, config.vad_max_segment_ms)
    return PreparedTranscriber(
        asr_model=asr_model,
        vad_model=vad_model,
        postprocessor=CantonesePostProcessor(),
        asr_backend=actual_backend,
        device=resolved_device,
        vad_device=vad_device,
        segment_batch_size=segment_batch_size,
        min_segment_ms=config.min_segment_ms,
        vad_max_segment_ms=config.vad_max_segment_ms,
        merge_target_segment_ms=config.merge_target_segment_ms,
        merge_max_segment_ms=config.merge_max_segment_ms,
        merge_max_gap_ms=config.merge_max_gap_ms,
        prep_workers=config.prep_workers,
        vad_workers=config.vad_workers,
        asr_prefetch_batches=config.asr_prefetch_batches,
        vad_cache_dir=config.vad_cache_dir,
        use_vad_cache=config.use_vad_cache,
        qwen_language=config.qwen_language,
        context_prompt=context_prompt,
        use_prompt=use_prompt,
    )


def collect_vad_segments(
    runtime: PreparedTranscriber,
    audio: Path,
    *,
    vad_model: Any | None = None,
) -> list[tuple[int, int]]:
    base_segments: list[tuple[int, int]] | None = None
    if runtime.use_vad_cache:
        cached = load_vad_cache(
            runtime.vad_cache_dir,
            audio,
            min_segment_ms=runtime.min_segment_ms,
            vad_max_segment_ms=runtime.vad_max_segment_ms,
        )
        if cached is not None:
            base_segments = cached

    if base_segments is None:
        active_vad_model = runtime.vad_model if vad_model is None else vad_model
        vad_res = active_vad_model.generate(input=str(audio))
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
        if runtime.asr_backend == "vllm":
            # Conservative default to avoid high host RAM pressure on long files.
            return max(4, min(8, runtime.segment_batch_size // 24))
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
        # Leave more room for vLLM + system processes to keep runs stable.
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
        if runtime.asr_backend == "vllm":
            # Keep prep moderate; oversubscription can hurt RAM/cache behavior.
            return max(2, min(4, cpu_budget // 4))
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
        if runtime.asr_backend == "vllm":
            # Cap VAD workers to avoid per-worker model memory blowups.
            return max(2, min(4, remaining_workers))
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
            context=runtime.context_prompt,
            language=runtime.qwen_language,
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
    # Keep enough per-file contribution so the queue fills quickly.
    per_file_enqueue = max(1, runtime.segment_batch_size // (resolved_active_files * 2))

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
        f"per_file_enqueue={per_file_enqueue}"
    )

    ordered_jobs = sort_jobs_by_duration(jobs)
    completed = 0
    failures: list[tuple[Path, str]] = []
    prepared_queue: Queue[PreparedJobItem] = Queue(maxsize=resolved_preload_files)
    producer_done = False
    feed_states: deque[FileBatchState] = deque()
    live_states: list[FileBatchState] = []
    segment_queue: list[SegmentTask] = []
    decoded_budget_cv = Condition()
    decoded_bytes_accounted = 0

    def reserve_decoded_budget(requested_bytes: int) -> int:
        nonlocal decoded_bytes_accounted
        reserved = max(1, int(requested_bytes))
        with decoded_budget_cv:
            while (
                decoded_bytes_accounted + reserved > decoded_budget_bytes
                and decoded_bytes_accounted > 0
            ):
                decoded_budget_cv.wait(timeout=0.2)
            decoded_bytes_accounted += reserved
        return reserved

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
        pending_vads: dict[Future[list[tuple[int, int]]], BatchJob] = {}
        pending_decodes: dict[
            Future[tuple[Any, int]],
            tuple[BatchJob, list[tuple[int, int]], int],
        ] = {}
        vad_backlog_limit = max(
            resolved_active_files + resolved_vad_workers,
            resolved_vad_workers * 2,
        )
        # Decoded audio payloads are the dominant host RAM consumer.
        # Never allow decode backlog to exceed preload window capacity.
        decode_backlog_limit = min(
            resolved_preload_files,
            max(
                resolved_active_files + resolved_prep_workers,
                resolved_prep_workers * 2,
            ),
        )
        vad_local = local()

        def collect_segments_for_job(job: BatchJob) -> list[tuple[int, int]]:
            if resolved_vad_workers <= 1:
                return collect_vad_segments(runtime, job.audio)
            worker_model = getattr(vad_local, "model", None)
            if worker_model is None:
                worker_model = build_vad_model(vad_worker_device, runtime.vad_max_segment_ms)
                setattr(vad_local, "model", worker_model)
            return collect_vad_segments(runtime, job.audio, vad_model=worker_model)

        def estimate_job_decode_bytes(segments: list[tuple[int, int]]) -> int:
            if not segments:
                return 1
            estimated_duration_ms = max(end_ms for _, end_ms in segments)
            return estimate_decoded_audio_bytes(estimated_duration_ms)

        def drain_decodes(*, block: bool) -> None:
            if not pending_decodes:
                return
            done, _ = wait(
                set(pending_decodes),
                timeout=None if block else 0.0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                return
            for future in done:
                job, segments, reserved_bytes = pending_decodes.pop(future)
                try:
                    audio_samples, audio_sr = future.result()
                except Exception as exc:  # noqa: BLE001
                    release_decoded_budget(reserved_bytes)
                    prepared_queue.put(
                        PreparedJobItem(
                            job=job,
                            failure=f"{type(exc).__name__}: {exc}",
                        )
                    )
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

        with (
            ThreadPoolExecutor(max_workers=resolved_prep_workers) as decode_pool,
            ThreadPoolExecutor(max_workers=resolved_vad_workers) as vad_pool,
        ):
            def drain_vads(*, block: bool) -> None:
                if not pending_vads:
                    return
                done, _ = wait(
                    set(pending_vads),
                    timeout=None if block else 0.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    return
                for future in done:
                    job = pending_vads.pop(future)
                    try:
                        segments = future.result()
                    except Exception as exc:  # noqa: BLE001
                        prepared_queue.put(
                            PreparedJobItem(
                                job=job,
                                failure=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue

                    if not segments:
                        prepared_queue.put(PreparedJobItem(job=job))
                        continue

                    reserved_bytes = reserve_decoded_budget(
                        estimate_job_decode_bytes(segments)
                    )
                    try:
                        decode_future = decode_pool.submit(load_audio_mono_16k, job.audio)
                    except Exception as exc:  # noqa: BLE001
                        release_decoded_budget(reserved_bytes)
                        prepared_queue.put(
                            PreparedJobItem(
                                job=job,
                                failure=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                    pending_decodes[decode_future] = (job, segments, reserved_bytes)

            for job in ordered_jobs:
                if not job.audio.is_file():
                    prepared_queue.put(
                        PreparedJobItem(
                            job=job,
                            failure=f"FileNotFoundError: Audio file not found: {job.audio}",
                        )
                    )
                    continue

                vad_future = vad_pool.submit(collect_segments_for_job, job)
                pending_vads[vad_future] = job
                drain_vads(block=False)
                drain_decodes(block=False)
                if len(pending_vads) >= vad_backlog_limit:
                    drain_vads(block=True)
                if len(pending_decodes) >= decode_backlog_limit:
                    drain_decodes(block=True)

            while pending_vads:
                drain_vads(block=True)
                drain_decodes(block=False)
            while pending_decodes:
                drain_decodes(block=True)
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

        rounds = len(feed_states)
        for _ in range(rounds):
            if len(segment_queue) >= segment_queue_capacity:
                break

            state = feed_states.popleft()
            if state.failed or state.done:
                continue

            if state.next_segment_idx >= len(state.segments):
                continue

            remaining = len(state.segments) - state.next_segment_idx
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

        return BatchBuildResult(
            batch_audio=batch_audio,
            batch_meta=batch_meta,
            failures=build_failures,
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
                    context=runtime.context_prompt,
                    language=runtime.qwen_language,
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

            batch_results = runtime.asr_model.transcribe(
                audio=batch_audio,
                context=runtime.context_prompt,
                language=runtime.qwen_language,
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
