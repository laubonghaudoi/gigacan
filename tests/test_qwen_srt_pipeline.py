from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gigacan.qwen_srt.batch import BatchJob
from gigacan.qwen_srt.pipeline import (
    BYTES_PER_GIB,
    FileBatchState,
    SegmentTask,
    estimate_decoded_audio_bytes,
    estimate_feature_frames,
    merge_small_vad_segments,
    resolve_decoded_audio_budget_bytes,
    resolve_per_file_enqueue,
    select_frame_aware_batch,
)


def _dummy_state() -> FileBatchState:
    return FileBatchState(
        job=BatchJob(audio=Path("dummy.opus"), output_srt=Path("dummy.srt")),
        segments=[],
    )


def test_estimate_feature_frames_uses_10ms_stride() -> None:
    assert estimate_feature_frames(0) == 1
    assert estimate_feature_frames(9) == 1
    assert estimate_feature_frames(10) == 1
    assert estimate_feature_frames(11) == 2


def test_merge_small_vad_segments_disabled_returns_input() -> None:
    segments = [(0, 500), (700, 1200)]
    merged = merge_small_vad_segments(
        segments,
        target_segment_ms=0,
        max_segment_ms=5000,
        max_gap_ms=200,
    )
    assert merged == segments


def test_merge_small_vad_segments_merges_short_neighbors_with_small_gap() -> None:
    segments = [(0, 1200), (1300, 2600), (8000, 9500)]
    merged = merge_small_vad_segments(
        segments,
        target_segment_ms=4000,
        max_segment_ms=12000,
        max_gap_ms=200,
    )
    assert merged == [(0, 2600), (8000, 9500)]


def test_merge_small_vad_segments_respects_max_duration_and_gap() -> None:
    segments = [(0, 3000), (3100, 7000), (7600, 9000)]
    merged = merge_small_vad_segments(
        segments,
        target_segment_ms=5000,
        max_segment_ms=6000,
        max_gap_ms=200,
    )
    # First two would exceed max_segment_ms if merged, third has larger gap.
    assert merged == segments


def test_select_frame_aware_batch_prefers_compact_short_segments() -> None:
    state = _dummy_state()
    queue = [
        SegmentTask(state=state, start_ms=0, end_ms=1000, duration_ms=1000, estimated_frames=100),
        SegmentTask(state=state, start_ms=0, end_ms=1100, duration_ms=1100, estimated_frames=110),
        SegmentTask(state=state, start_ms=0, end_ms=1200, duration_ms=1200, estimated_frames=120),
        SegmentTask(state=state, start_ms=0, end_ms=7000, duration_ms=7000, estimated_frames=700),
        SegmentTask(state=state, start_ms=0, end_ms=7100, duration_ms=7100, estimated_frames=710),
    ]
    selected = select_frame_aware_batch(queue, batch_size=3)

    assert len(selected) == 3
    assert sorted(task.duration_ms for task in selected) == [1000, 1100, 1200]
    assert len(queue) == 2


def test_select_frame_aware_batch_enforces_batch_size_limit() -> None:
    state = _dummy_state()
    queue = [
        SegmentTask(
            state=state,
            start_ms=0,
            end_ms=1000 + idx * 20,
            duration_ms=1000 + idx * 20,
            estimated_frames=100 + idx * 2,
        )
        for idx in range(10)
    ]

    selected = select_frame_aware_batch(queue, batch_size=4)
    assert len(selected) <= 4
    assert len(selected) + len(queue) == 10


def test_estimate_decoded_audio_bytes_has_overhead() -> None:
    one_second = estimate_decoded_audio_bytes(1000)
    assert one_second > 16_000 * 4


def test_resolve_per_file_enqueue_scales_up_when_feeders_shrink() -> None:
    # Full active set: keep fair contribution.
    full = resolve_per_file_enqueue(
        segment_batch_size=1024,
        configured_active_files=16,
        current_feed_states=16,
    )
    # Tail phase with one feeder: push aggressively to keep batches dense.
    tail = resolve_per_file_enqueue(
        segment_batch_size=1024,
        configured_active_files=16,
        current_feed_states=1,
    )
    assert full == 64
    assert tail > full
    assert tail == 256


def test_resolve_decoded_audio_budget_bytes_honors_explicit_gib() -> None:
    runtime = SimpleNamespace(device="cuda:0")
    budget = resolve_decoded_audio_budget_bytes(runtime, 3.5)
    assert budget == int(3.5 * BYTES_PER_GIB)


def test_resolve_decoded_audio_budget_bytes_auto_cuda_cap(monkeypatch) -> None:
    runtime = SimpleNamespace(device="cuda:0")
    monkeypatch.setattr(
        "gigacan.qwen_srt.pipeline.try_get_total_memory_bytes",
        lambda: 64 * BYTES_PER_GIB,
    )
    budget = resolve_decoded_audio_budget_bytes(runtime, 0.0)
    assert budget == 12 * BYTES_PER_GIB
