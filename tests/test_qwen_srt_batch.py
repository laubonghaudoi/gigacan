from __future__ import annotations

import csv
from pathlib import Path

from gigacan.qwen_srt import batch
from gigacan.qwen_srt.batch import BatchJob, sort_jobs_by_duration


def _touch(path: Path, size_bytes: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)


def _write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["id", "audio", "duration", "duration_seconds", "url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_legco_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "url",
        "title",
        "description",
        "publish_date",
        "duration",
        "downloaded",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_sort_jobs_by_duration_uses_metadata_hints_without_ffprobe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _touch(Path("download/2025/a.opus"), size_bytes=10)
    _touch(Path("download/2025/b.opus"), size_bytes=10)
    _touch(Path("download/2025/c.opus"), size_bytes=10)

    _write_metadata_csv(
        Path("metadata.csv"),
        [
            {
                "id": "a",
                "audio": "download/2025/a.opus",
                "duration": "00:01:40",
                "duration_seconds": "100",
                "url": "https://example.com/a",
            },
            {
                "id": "b",
                "audio": "download/2025/b.opus",
                "duration": "00:00:10",
                "duration_seconds": "10",
                "url": "https://example.com/b",
            },
            {
                "id": "c",
                "audio": "download/2025/c.opus",
                "duration": "00:01:10",
                "duration_seconds": "70",
                "url": "https://example.com/c",
            },
        ],
    )

    def fail_probe(_audio: Path) -> float:
        raise AssertionError("ffprobe should not be used when metadata covers all jobs")

    monkeypatch.setattr(batch, "probe_audio_duration_seconds", fail_probe)

    jobs = [
        BatchJob(audio=Path("download/2025/a.opus"), output_srt=Path("out/a.srt")),
        BatchJob(audio=Path("download/2025/b.opus"), output_srt=Path("out/b.srt")),
        BatchJob(audio=Path("download/2025/c.opus"), output_srt=Path("out/c.srt")),
    ]
    ordered = sort_jobs_by_duration(jobs)
    assert [job.audio.name for job in ordered] == ["a.opus", "b.opus", "c.opus"]


def test_sort_jobs_by_duration_uses_legco_duration_hints_without_ffprobe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _touch(Path("download/2025/a.opus"), size_bytes=10)
    _touch(Path("download/2025/b.opus"), size_bytes=10)
    _touch(Path("download/2025/c.opus"), size_bytes=10)

    _write_legco_csv(
        Path("legco.csv"),
        [
            {
                "url": "https://www.youtube.com/watch?v=a",
                "title": "A",
                "description": "",
                "publish_date": "2025-01-01",
                "duration": "00:01:40",
                "downloaded": "True",
            },
            {
                "url": "https://www.youtube.com/watch?v=b",
                "title": "B",
                "description": "",
                "publish_date": "2025-01-01",
                "duration": "00:00:10",
                "downloaded": "True",
            },
            {
                "url": "https://www.youtube.com/watch?v=c",
                "title": "C",
                "description": "",
                "publish_date": "2025-01-01",
                "duration": "00:01:10",
                "downloaded": "True",
            },
        ],
    )

    def fail_probe(_audio: Path) -> float:
        raise AssertionError("ffprobe should not be used when legco.csv covers all jobs")

    monkeypatch.setattr(batch, "probe_audio_duration_seconds", fail_probe)

    jobs = [
        BatchJob(audio=Path("download/2025/a.opus"), output_srt=Path("out/a.srt")),
        BatchJob(audio=Path("download/2025/b.opus"), output_srt=Path("out/b.srt")),
        BatchJob(audio=Path("download/2025/c.opus"), output_srt=Path("out/c.srt")),
    ]
    ordered = sort_jobs_by_duration(jobs)
    assert [job.audio.name for job in ordered] == ["a.opus", "b.opus", "c.opus"]


def test_sort_jobs_by_duration_falls_back_to_ffprobe_for_small_missing_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _touch(Path("download/a.opus"), size_bytes=100)
    _touch(Path("download/b.opus"), size_bytes=100)
    _touch(Path("download/c.opus"), size_bytes=100)

    calls: list[str] = []
    duration_by_name = {
        "a.opus": 100.0,
        "b.opus": 10.0,
        "c.opus": 70.0,
    }

    def fake_probe(audio: Path) -> float:
        calls.append(audio.name)
        return duration_by_name[audio.name]

    monkeypatch.setattr(batch, "probe_audio_duration_seconds", fake_probe)
    monkeypatch.setattr(batch, "MAX_FFPROBE_FALLBACK_JOBS", 10)

    jobs = [
        BatchJob(audio=Path("download/a.opus"), output_srt=Path("out/a.srt")),
        BatchJob(audio=Path("download/b.opus"), output_srt=Path("out/b.srt")),
        BatchJob(audio=Path("download/c.opus"), output_srt=Path("out/c.srt")),
    ]
    ordered = sort_jobs_by_duration(jobs)
    assert sorted(calls) == ["a.opus", "b.opus", "c.opus"]
    assert [job.audio.name for job in ordered] == ["a.opus", "b.opus", "c.opus"]


def test_sort_jobs_by_duration_uses_filesize_estimate_when_missing_set_is_large(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _touch(Path("download/a.opus"), size_bytes=8_000)
    _touch(Path("download/b.opus"), size_bytes=1_000)
    _touch(Path("download/c.opus"), size_bytes=5_000)

    def fail_probe(_audio: Path) -> float:
        raise AssertionError("ffprobe should be skipped for large unresolved sets")

    monkeypatch.setattr(batch, "probe_audio_duration_seconds", fail_probe)
    monkeypatch.setattr(batch, "MAX_FFPROBE_FALLBACK_JOBS", 2)

    jobs = [
        BatchJob(audio=Path("download/a.opus"), output_srt=Path("out/a.srt")),
        BatchJob(audio=Path("download/b.opus"), output_srt=Path("out/b.srt")),
        BatchJob(audio=Path("download/c.opus"), output_srt=Path("out/c.srt")),
    ]
    ordered = sort_jobs_by_duration(jobs)
    assert [job.audio.name for job in ordered] == ["a.opus", "b.opus", "c.opus"]
