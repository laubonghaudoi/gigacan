from __future__ import annotations

# pyright: reportMissingTypeStubs=false

from collections.abc import Hashable
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas import DataFrame


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


def _load_downloaded_ids(download_dir: Path) -> set[str]:
    """Return the set of video IDs inferred from ``download_dir``."""

    downloaded_ids: set[str] = set()
    for path in download_dir.rglob("*.opus"):
        if path.is_file():
            downloaded_ids.add(path.stem)
    return downloaded_ids


def _is_marked_downloaded(value: object) -> bool:
    """Interpret the ``downloaded`` column values as booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _ensure_download_column(df: DataFrame) -> None:
    if "downloaded" not in df.columns:
        df["downloaded"] = False


def scan_and_update_progress(
    download_dir: Path = DOWNLOAD_DIR, csv_path: Path = CSV_FILE
) -> None:
    """
    Scan the download directory and update the metadata CSV to match reality.
    """

    print(f"Scanning for downloaded files in '{download_dir}'...")

    if not download_dir.is_dir():
        print(f"Error: Download directory '{download_dir}' not found.")
        return

    downloaded_ids = _load_downloaded_ids(download_dir)
    print(
        f"Found {len(downloaded_ids)} downloaded audio files across all subdirectories."
    )

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return
    except (pd.errors.EmptyDataError, OSError) as exc:
        print(f"Error reading CSV file '{csv_path}': {exc}")
        return

    _ensure_download_column(df)

    updates_made = 0
    missing_files_count = 0

    records = df.to_dict(orient="records")
    indices = cast(list[Hashable], list(df.index))

    for index_label, record in zip(indices, records):
        video_id = get_video_id(cast(object, record.get("url")))
        is_marked = _is_marked_downloaded(cast(object, record.get("downloaded")))

        if is_marked and (video_id not in downloaded_ids):
            missing_files_count += 1
        elif not is_marked and video_id and (video_id in downloaded_ids):
            df.loc[index_label, "downloaded"] = True
            updates_made += 1

    if updates_made:
        print(f"Updating {updates_made} rows in the CSV to 'downloaded: True'.")
        try:
            df.to_csv(csv_path, index=False)
            print(f"Successfully updated '{csv_path}'.")
        except (OSError, IOError) as exc:
            print(f"Error saving updated CSV file: {exc}")
    else:
        print("No updates needed. CSV file is already in sync with the download folder.")

    if missing_files_count:
        warning_message = (
            f"Warning: Found {missing_files_count} files marked as 'True' in the CSV "
            + "but not found in the download folder."
        )
        print(warning_message)


if __name__ == "__main__":
    scan_and_update_progress()
