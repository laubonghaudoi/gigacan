from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.batch import (
    build_batch_jobs,
    discover_audio_files,
    filter_audio_files_by_year,
    output_srt_for_audio,
    sort_jobs_by_duration,
    BatchJob,
)


def test_discover_audio_files_filters_and_sorts(tmp_path: Path) -> None:
    root = tmp_path / "download"
    (root / "2025").mkdir(parents=True)
    (root / "2024").mkdir(parents=True)
    (root / "2025" / "b.opus").write_text("", encoding="utf-8")
    (root / "2024" / "a.mp3").write_text("", encoding="utf-8")
    (root / "2025" / "note.txt").write_text("not audio", encoding="utf-8")

    files = discover_audio_files(root)

    assert files == [root / "2024" / "a.mp3", root / "2025" / "b.opus"]


def test_output_srt_for_audio_mirrors_directory_structure(tmp_path: Path) -> None:
    audio_root = tmp_path / "download"
    output_root = tmp_path / "transcriptions"
    audio = audio_root / "2025" / "xyz.opus"
    audio.parent.mkdir(parents=True)
    audio.write_text("", encoding="utf-8")

    output_srt = output_srt_for_audio(audio, audio_root, output_root)

    assert output_srt == output_root / "2025" / "xyz.srt"


def test_build_batch_jobs_skips_existing_when_not_overwrite(tmp_path: Path) -> None:
    audio_root = tmp_path / "download"
    output_root = tmp_path / "transcriptions"
    audio_a = audio_root / "2025" / "a.opus"
    audio_b = audio_root / "2025" / "b.opus"
    audio_a.parent.mkdir(parents=True)
    audio_a.write_text("", encoding="utf-8")
    audio_b.write_text("", encoding="utf-8")
    existing = output_root / "2025" / "a.srt"
    existing.parent.mkdir(parents=True)
    existing.write_text("", encoding="utf-8")

    jobs, skipped = build_batch_jobs(
        [audio_a, audio_b],
        audio_root,
        output_root,
        overwrite=False,
    )

    assert skipped == 1
    assert [job.audio for job in jobs] == [audio_b]
    assert [job.output_srt for job in jobs] == [output_root / "2025" / "b.srt"]


def test_filter_audio_files_by_year_matches_first_path_component(tmp_path: Path) -> None:
    audio_root = tmp_path / "download"
    a2024 = audio_root / "2024" / "a.opus"
    a2025 = audio_root / "2025" / "b.opus"
    nested2025 = audio_root / "2025" / "nested" / "c.opus"
    for path in [a2024, a2025, nested2025]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    filtered = filter_audio_files_by_year(
        [a2024, a2025, nested2025],
        audio_root,
        "2025",
    )

    assert filtered == [a2025, nested2025]


def test_sort_jobs_by_duration_interleaves_long_and_short_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "download" / "2025"
    root.mkdir(parents=True)
    short = root / "short.opus"
    mid = root / "mid.opus"
    long = root / "long.opus"
    for audio in [short, mid, long]:
        audio.write_text("", encoding="utf-8")

    jobs = [
        BatchJob(audio=short, output_srt=short.with_suffix(".srt")),
        BatchJob(audio=mid, output_srt=mid.with_suffix(".srt")),
        BatchJob(audio=long, output_srt=long.with_suffix(".srt")),
    ]

    durations = {
        short: 10.0,
        mid: 20.0,
        long: 30.0,
    }

    monkeypatch.setattr(
        "gigacan.qwen_srt.batch.probe_audio_duration_seconds",
        lambda audio: durations[audio],
    )

    ordered = sort_jobs_by_duration(jobs)

    assert [job.audio for job in ordered] == [long, short, mid]
