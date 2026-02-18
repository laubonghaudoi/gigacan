from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
    }
)


@dataclass(slots=True, frozen=True)
class BatchJob:
    audio: Path
    output_srt: Path


def discover_audio_files(audio_root: Path) -> list[Path]:
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")
    return sorted(
        path
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def filter_audio_files_by_year(
    audio_files: list[Path],
    audio_root: Path,
    year: str,
) -> list[Path]:
    year_str = year.strip()
    filtered: list[Path] = []
    for audio in audio_files:
        relative_parts = audio.relative_to(audio_root).parts
        if relative_parts and relative_parts[0] == year_str:
            filtered.append(audio)
    return filtered


def output_srt_for_audio(audio: Path, audio_root: Path, output_root: Path) -> Path:
    relative_audio = audio.relative_to(audio_root)
    return (output_root / relative_audio).with_suffix(".srt")


def build_batch_jobs(
    audio_files: list[Path],
    audio_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> tuple[list[BatchJob], int]:
    jobs: list[BatchJob] = []
    skipped = 0
    for audio in audio_files:
        output_srt = output_srt_for_audio(audio, audio_root, output_root)
        if output_srt.exists() and not overwrite:
            skipped += 1
            continue
        jobs.append(BatchJob(audio=audio, output_srt=output_srt))
    return jobs, skipped


def probe_audio_duration_seconds(audio: Path) -> float:
    """Probe audio duration with ffprobe. Returns 0.0 when probing fails."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = proc.stdout.strip()
        if not value:
            return 0.0
        duration = float(value)
        return duration if duration > 0 else 0.0
    except Exception:
        return 0.0


def sort_jobs_by_duration(jobs: Sequence[BatchJob]) -> list[BatchJob]:
    """Interleave long and short jobs to smooth CPU/RAM pressure across files."""
    jobs_with_duration = [
        (probe_audio_duration_seconds(job.audio), index, job)
        for index, job in enumerate(jobs)
    ]
    jobs_with_duration.sort(key=lambda item: (-item[0], item[1]))

    ordered: list[BatchJob] = []
    left = 0
    right = len(jobs_with_duration) - 1
    while left <= right:
        ordered.append(jobs_with_duration[left][2])
        left += 1
        if left <= right:
            ordered.append(jobs_with_duration[right][2])
            right -= 1
    return ordered
