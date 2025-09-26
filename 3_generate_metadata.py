#!/usr/bin/env python3
"""Convert playlist metadata to the Hugging Face dataset manifest format."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse

import pandas as pd
from tqdm import tqdm


DEFAULT_SOURCE = Path("legco.csv")
DEFAULT_OUTPUT = Path("metadata.csv")
DEFAULT_DOWNLOAD_DIR = Path("download")


@dataclass
class MetadataRow:
    """Container for a single Hugging Face metadata entry."""

    id: str
    audio: str
    title: str
    description: str
    publish_date: str
    duration: str
    duration_seconds: int


@dataclass
class CandidateEntry:
    video_id: str
    audio_path: Path
    publish_date: str
    title: str
    description: str


def parse_args() -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the source metadata CSV into the Hugging Face metadata format. "
            "Only rows with downloaded audio and a discoverable .opus file are exported."
        )
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source CSV file to read (default: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        '--download-dir',
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Directory that stores downloaded audio (default: {DEFAULT_DOWNLOAD_DIR})"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--jobs',
        type=_parse_jobs,
        default=None,
        help=(
            "Number of worker processes to use when probing durations (default: auto). "
            "Use 0 to auto-detect, 1 to force sequential behaviour."
        ),
    )
    return parser.parse_args()


def _parse_jobs(value: str) -> int:
    jobs = int(value)
    if jobs < 0:
        raise argparse.ArgumentTypeError('jobs must be >= 0')
    return jobs


def get_video_id(url: str | float | None) -> str | None:
    """Extract a YouTube video ID from ``url`` when possible."""
    if url is None or pd.isna(url):
        return None
    url_str = str(url)
    if 'youtu.be' in url_str:
        return url_str.rstrip('/').split('/')[-1].split('?')[0]
    try:
        parsed = urlparse(url_str)
    except ValueError:
        return None
    if parsed.hostname and parsed.hostname.replace('www.', '') == 'youtube.com':
        video_ids = parse_qs(parsed.query).get('v')
        if video_ids:
            return video_ids[0]
    return None


def normalise_downloaded(value: object) -> bool:
    """Return ``True`` when ``value`` indicates the audio has been downloaded."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {'1', 'true', 'yes', 'y'}
    return False


def resolve_audio_path(download_dir: Path, video_id: str, publish_date: object) -> Path | None:
    """Attempt to locate the on-disk .opus file for ``video_id``."""
    candidates: list[Path] = []
    year: str | None = None
    if publish_date is not None and not pd.isna(publish_date):
        year = str(publish_date)[:4]
        if year.isdigit():
            candidates.append(download_dir / year / f'{video_id}.opus')
    candidates.append(download_dir / f'{video_id}.opus')
    # Some downloads might still be in nested folders such as ``download/<id>/<id>.opus``
    candidates.append(download_dir / video_id / f'{video_id}.opus')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def probe_duration_seconds(audio_path: Path) -> int | None:
    """Return the rounded duration (in seconds) of ``audio_path`` using ffprobe."""
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(audio_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required to determine audio durations. Install FFmpeg and retry."
        ) from exc
    except subprocess.CalledProcessError:
        return None

    try:
        seconds = float(output)
    except (TypeError, ValueError):
        return None

    if seconds <= 0:
        return None
    return int(round(seconds))


def format_seconds_hms(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def determine_worker_count(jobs: int | None) -> int:
    if jobs is None or jobs == 0:
        return max(1, os.cpu_count() or 1)
    return jobs


def probe_audio_durations(paths: Sequence[Path], jobs: int | None) -> tuple[dict[Path, tuple[int, str]], list[Path]]:
    if not paths:
        return {}, []

    worker_count = determine_worker_count(jobs)
    results: dict[Path, tuple[int, str]] = {}
    failures: list[Path] = []

    progress = tqdm(
        total=len(paths),
        desc="Probing durations",
        unit="file",
        leave=False,
        disable=not sys.stdout.isatty(),
    )

    def _record_result(path: Path, seconds: int | None) -> None:
        if seconds is None:
            failures.append(path)
        else:
            results[path] = (seconds, format_seconds_hms(seconds))
        progress.update(1)

    if worker_count == 1:
        for path in paths:
            try:
                seconds = probe_duration_seconds(path)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            except Exception:
                failures.append(path)
                continue
            _record_result(path, seconds)
        progress.close()
        return results, failures

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_probe_path_seconds, str(path)): path for path in paths
        }
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                seconds = future.result()
            except RuntimeError as exc:
                # Propagate ffprobe availability issues immediately.
                executor.shutdown(cancel_futures=True)
                raise SystemExit(str(exc)) from exc
            except Exception:
                failures.append(path)
                continue
            _record_result(path, seconds)

    progress.close()

    return results, failures


def _probe_path_seconds(path_str: str) -> int | None:
    return probe_duration_seconds(Path(path_str))


def build_rows(
    df: pd.DataFrame,
    download_dir: Path,
    jobs: int | None,
) -> tuple[list[MetadataRow], dict[str, list[str]]]:
    """Create metadata rows and collect diagnostics about skipped entries."""
    rows: list[MetadataRow] = []
    diagnostics = {
        'missing_video_id': [],
        'not_downloaded': [],
        'missing_audio_file': [],
        'duration_probe_failed': [],
    }

    candidates: list[CandidateEntry] = []
    paths_for_probe: list[Path] = []
    video_ids_by_path: dict[Path, list[str]] = {}

    for record in df.itertuples(index=False):
        url = getattr(record, 'url', None)
        video_id = get_video_id(url)
        if not video_id:
            diagnostics['missing_video_id'].append(str(url))
            continue

        downloaded_value = getattr(record, 'downloaded', None)
        if not normalise_downloaded(downloaded_value):
            diagnostics['not_downloaded'].append(video_id)
            continue

        publish_date_value = getattr(record, 'publish_date', None)
        audio_path = resolve_audio_path(download_dir, video_id, publish_date_value)
        if audio_path is None:
            diagnostics['missing_audio_file'].append(video_id)
            continue

        if audio_path not in video_ids_by_path:
            paths_for_probe.append(audio_path)
        video_ids_by_path.setdefault(audio_path, []).append(video_id)

        candidates.append(
            CandidateEntry(
                video_id=video_id,
                audio_path=audio_path,
                publish_date=str(publish_date_value) if publish_date_value is not None else '',
                title=str(getattr(record, 'title', '') or ''),
                description=str(getattr(record, 'description', '') or ''),
            )
        )

    duration_cache, failed_paths = probe_audio_durations(paths_for_probe, jobs)

    for failed_path in failed_paths:
        diagnostics['duration_probe_failed'].extend(video_ids_by_path.get(failed_path, []))

    for candidate in candidates:
        cached = duration_cache.get(candidate.audio_path)
        if cached is None:
            continue
        duration_seconds, duration_display = cached
        rows.append(
            MetadataRow(
                id=candidate.video_id,
                audio=candidate.audio_path.as_posix(),
                title=candidate.title,
                description=candidate.description,
                publish_date=candidate.publish_date,
                duration=duration_display,
                duration_seconds=duration_seconds,
            )
        )
    return rows, diagnostics


def report_diagnostics(rows: Sequence[MetadataRow], diagnostics: dict[str, list[str]]) -> None:
    """Emit a short summary of the conversion process."""
    print(f"✓ Prepared {len(rows)} metadata entries.")
    for key, items in diagnostics.items():
        if not items:
            continue
        print(f"- Skipped {len(items)} entries due to {key.replace('_', ' ')}")


def main() -> None:
    """Entry point for the metadata conversion utility."""
    args = parse_args()
    source = args.source
    download_dir = args.download_dir
    output = args.output

    if not source.exists():
        raise SystemExit(f"Source CSV not found: {source}")
    if not download_dir.exists():
        raise SystemExit(f"Download directory not found: {download_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    rows, diagnostics = build_rows(df, download_dir, args.jobs)
    if not rows:
        raise SystemExit("No metadata rows generated; check diagnostics and input files.")

    data = [row.__dict__ for row in rows]
    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(output, index=False)

    report_diagnostics(rows, diagnostics)
    print(f"✓ Wrote metadata CSV to {output}")


if __name__ == '__main__':
    main()
