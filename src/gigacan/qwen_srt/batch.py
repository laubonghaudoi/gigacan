from __future__ import annotations

import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse


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

DEFAULT_METADATA_DURATION_INDEX_CSV = "metadata.csv"
DEFAULT_LEGCO_DURATION_INDEX_CSV = "legco.csv"
ESTIMATED_AUDIO_BYTES_PER_SECOND = 8_000.0
MAX_FFPROBE_FALLBACK_JOBS = 512
MAX_FFPROBE_WORKERS = 16


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


def _audio_lookup_keys(audio: Path) -> tuple[str, ...]:
    keys: set[str] = set()
    keys.add(str(audio))
    keys.add(audio.as_posix())
    try:
        resolved = audio.resolve()
        keys.add(str(resolved))
        keys.add(resolved.as_posix())
    except Exception:
        pass
    if not audio.is_absolute():
        try:
            cwd_resolved = (Path.cwd() / audio).resolve()
            keys.add(str(cwd_resolved))
            keys.add(cwd_resolved.as_posix())
        except Exception:
            pass
    return tuple(keys)


def _safe_positive_float(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _safe_positive_duration_seconds(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return None

    direct = _safe_positive_float(value)
    if direct is not None:
        return direct

    parts = value.split(":")
    if not parts:
        return None
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
            total = float(hours * 3600 + minutes * 60 + seconds)
        elif len(parts) == 2:
            minutes, seconds = (int(part) for part in parts)
            total = float(minutes * 60 + seconds)
        else:
            return None
    except ValueError:
        return None
    if total <= 0:
        return None
    return total


def _extract_video_id_from_url(raw_url: str) -> str | None:
    url = raw_url.strip()
    if not url:
        return None
    if "youtu.be/" in url:
        tail = url.rstrip("/").split("/")[-1]
        candidate = tail.split("?")[0].strip()
        return candidate or None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    candidate = (parse_qs(parsed.query).get("v") or [""])[0].strip()
    return candidate or None


def _load_metadata_duration_hints(
    path_map: dict[str, float],
    id_map: dict[str, float],
) -> None:
    csv_path = Path(DEFAULT_METADATA_DURATION_INDEX_CSV)
    if not csv_path.is_file():
        return

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration_raw = str(row.get("duration_seconds") or "")
            duration = _safe_positive_duration_seconds(duration_raw)
            if duration is None:
                continue

            audio_raw = str(row.get("audio") or "").strip()
            if audio_raw:
                for key in _audio_lookup_keys(Path(audio_raw)):
                    path_map.setdefault(key, duration)

            id_raw = str(row.get("id") or "").strip()
            if id_raw:
                id_map.setdefault(id_raw, duration)


def _load_legco_duration_hints(id_map: dict[str, float]) -> None:
    csv_path = Path(DEFAULT_LEGCO_DURATION_INDEX_CSV)
    if not csv_path.is_file():
        return

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration = _safe_positive_duration_seconds(
                str(row.get("duration_seconds") or "")
            )
            if duration is None:
                duration = _safe_positive_duration_seconds(str(row.get("duration") or ""))
            if duration is None:
                continue

            video_id = str(row.get("id") or "").strip()
            if not video_id:
                video_id = _extract_video_id_from_url(str(row.get("url") or "")) or ""
            if video_id:
                id_map.setdefault(video_id, duration)


def _load_duration_hints() -> tuple[dict[str, float], dict[str, float]]:
    path_map: dict[str, float] = {}
    id_map: dict[str, float] = {}

    try:
        _load_metadata_duration_hints(path_map, id_map)
        _load_legco_duration_hints(id_map)
    except Exception:
        return {}, {}

    return path_map, id_map


def _lookup_duration_hint_seconds(
    audio: Path,
    path_map: dict[str, float],
    id_map: dict[str, float],
) -> float | None:
    for key in _audio_lookup_keys(audio):
        duration = path_map.get(key)
        if duration is not None:
            return duration

    stem_duration = id_map.get(audio.stem)
    if stem_duration is not None:
        return stem_duration
    return None


def estimate_duration_from_filesize_seconds(audio: Path) -> float:
    try:
        size_bytes = int(audio.stat().st_size)
    except Exception:
        return 0.0
    if size_bytes <= 0:
        return 0.0
    return float(size_bytes) / ESTIMATED_AUDIO_BYTES_PER_SECOND


def probe_durations_parallel(
    jobs: Sequence[tuple[int, BatchJob]],
) -> dict[int, float]:
    if not jobs:
        return {}

    workers = min(
        MAX_FFPROBE_WORKERS,
        max(1, os.cpu_count() or 1),
        len(jobs),
    )
    durations: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(probe_audio_duration_seconds, job.audio): index
            for index, job in jobs
        }
        for future, index in future_to_index.items():
            try:
                durations[index] = max(0.0, float(future.result()))
            except Exception:
                durations[index] = 0.0
    return durations


def sort_jobs_by_duration(jobs: Sequence[BatchJob]) -> list[BatchJob]:
    """Interleave long and short jobs to smooth CPU/RAM pressure across files."""
    jobs_list = list(jobs)
    if not jobs_list:
        return []

    path_map, id_map = _load_duration_hints()
    hint_hits = 0

    job_durations: list[float] = []
    unresolved: list[tuple[int, BatchJob]] = []
    for index, job in enumerate(jobs_list):
        hinted = _lookup_duration_hint_seconds(job.audio, path_map, id_map)
        if hinted is not None:
            hint_hits += 1
            job_durations.append(hinted)
            continue
        job_durations.append(0.0)
        unresolved.append((index, job))

    ffprobe_hits = 0
    if unresolved and len(unresolved) <= MAX_FFPROBE_FALLBACK_JOBS:
        probed = probe_durations_parallel(unresolved)
        for index, duration in probed.items():
            if duration > 0:
                ffprobe_hits += 1
                job_durations[index] = duration

    filesize_hits = 0
    for index, job in unresolved:
        if job_durations[index] > 0:
            continue
        estimated = estimate_duration_from_filesize_seconds(job.audio)
        if estimated > 0:
            filesize_hits += 1
            job_durations[index] = estimated

    if unresolved:
        print(
            "Duration ordering hints: "
            f"hints={hint_hits}, "
            f"ffprobe={ffprobe_hits}, "
            f"filesize={filesize_hits}, "
            f"missing={max(0, len(unresolved) - ffprobe_hits - filesize_hits)}"
        )

    jobs_with_duration = [
        (job_durations[index], index, job)
        for index, job in enumerate(jobs_list)
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
