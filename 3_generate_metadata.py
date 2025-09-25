#!/usr/bin/env python3
"""Convert playlist metadata to the Hugging Face dataset manifest format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse

import pandas as pd


DEFAULT_SOURCE = Path('legco_20250920.csv')
DEFAULT_OUTPUT = Path('metadata.csv')
DEFAULT_DOWNLOAD_DIR = Path('download')


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
    return parser.parse_args()


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


def parse_duration_to_seconds(duration: str | float | None) -> int | None:
    """Convert HH:MM:SS or MM:SS strings into total seconds."""
    if duration is None or pd.isna(duration):
        return None
    parts = str(duration).split(':')
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        return int(parts[0])
    except (TypeError, ValueError):
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


def build_rows(df: pd.DataFrame, download_dir: Path) -> tuple[list[MetadataRow], dict[str, list[str]]]:
    """Create metadata rows and collect diagnostics about skipped entries."""
    rows: list[MetadataRow] = []
    diagnostics = {
        'missing_video_id': [],
        'not_downloaded': [],
        'missing_audio_file': [],
        'missing_duration': [],
    }

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

        publish_date = getattr(record, 'publish_date', None)
        audio_path = resolve_audio_path(download_dir, video_id, publish_date)
        if audio_path is None:
            diagnostics['missing_audio_file'].append(video_id)
            continue

        duration_str = getattr(record, 'duration', None)
        duration_seconds = parse_duration_to_seconds(duration_str)
        if duration_seconds is None:
            diagnostics['missing_duration'].append(video_id)
            continue

        rows.append(
            MetadataRow(
                id=video_id,
                audio=audio_path.as_posix(),
                title=str(getattr(record, 'title', '') or ''),
                description=str(getattr(record, 'description', '') or ''),
                publish_date=str(publish_date) if publish_date is not None else '',
                duration=str(duration_str) if duration_str is not None else '',
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
    rows, diagnostics = build_rows(df, download_dir)
    if not rows:
        raise SystemExit("No metadata rows generated; check diagnostics and input files.")

    data = [row.__dict__ for row in rows]
    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(output, index=False)

    report_diagnostics(rows, diagnostics)
    print(f"✓ Wrote metadata CSV to {output}")


if __name__ == '__main__':
    main()
