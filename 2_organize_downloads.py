from __future__ import annotations

# pyright: reportMissingTypeStubs=false

import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas import DataFrame
from typing import cast


# --- CONFIGURATION ---
CSV_FILE = Path("legco.csv")
DOWNLOAD_DIR = Path("download")
# --- END CONFIGURATION ---


def get_video_id(url: object) -> str | None:
    """Extract the YouTube video ID from ``url``."""

    if not isinstance(url, str):
        return None

    if "youtu.be" in url:
        parts = url.rstrip("/").split("/")
        if parts:
            candidate = parts[-1].split("?")[0]
            return candidate or None
        return None

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"www.youtube.com", "youtube.com"}:
        return None

    candidates = parse_qs(parsed.query).get("v")
    if not candidates:
        return None

    return candidates[0] or None


def _build_video_year_map(df: DataFrame) -> dict[str, str]:
    """Return a mapping from video ID to publication year."""

    mapping: dict[str, str] = {}
    records = df.to_dict(orient="records")

    for record in records:
        video_id = get_video_id(cast(object, record.get("url")))
        publish_date = record.get("publish_date")

        if video_id and isinstance(publish_date, str) and len(publish_date) >= 4:
            mapping[video_id] = publish_date.split("-")[0]

    return mapping


def organise_downloads_by_year(
    csv_path: Path = CSV_FILE, download_dir: Path = DOWNLOAD_DIR
) -> None:
    """
    Move every downloaded ``.opus`` file into a year-based subdirectory.
    """

    print(f"Reading video metadata from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return
    except (pd.errors.EmptyDataError, OSError) as exc:
        print(f"Error reading CSV file: {exc}")
        return

    video_id_to_year = _build_video_year_map(df)
    print(f"Created a map for {len(video_id_to_year)} videos.")

    if not download_dir.is_dir():
        print(f"Error: Download directory '{download_dir}' not found.")
        return

    files_to_organise = [
        path for path in download_dir.iterdir() if path.suffix == ".opus" and path.is_file()
    ]

    if not files_to_organise:
        print("No .opus files found in the download directory to organise.")
        return

    moved_count = 0
    unmapped_count = 0
    print(f"Starting to organise {len(files_to_organise)} files...")

    for source_path in files_to_organise:
        video_id = source_path.stem
        year = video_id_to_year.get(video_id)

        if not year:
            print(f"Warning: No upload year found for video ID '{video_id}'. Skipping.")
            unmapped_count += 1
            continue

        year_dir = download_dir / year
        destination = year_dir / source_path.name

        try:
            year_dir.mkdir(parents=True, exist_ok=True)
            _ = shutil.move(str(source_path), str(destination))
            moved_count += 1
        except (OSError, shutil.Error) as exc:
            print(f"Error moving '{source_path.name}': {exc}")

    print("\n--- Organisation Complete ---")
    print(f"Successfully moved {moved_count} files.")
    if unmapped_count:
        print(f"{unmapped_count} files were skipped as they could not be mapped to a year.")


if __name__ == "__main__":
    organise_downloads_by_year()
