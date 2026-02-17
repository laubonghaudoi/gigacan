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
SUBTITLE_DIR = Path("subtitle")
TARGET_SUBTITLE_LANGS = {"yue-hant", "yue"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ttml", ".lrc", ".json3"}
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


def _normalise_lang(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _is_target_subtitle_lang(lang: str) -> bool:
    lang_norm = _normalise_lang(lang)
    if lang_norm in TARGET_SUBTITLE_LANGS:
        return True
    if any(
        lang_norm.startswith(target + "-") for target in sorted(TARGET_SUBTITLE_LANGS)
    ):
        return True
    return False


def _load_subtitle_ids(subtitle_dir: Path) -> set[str]:
    """Return video IDs that already have yue/yue-Hant subtitle files."""

    subtitle_ids: set[str] = set()
    for path in subtitle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue

        parts = path.name.split(".")
        if len(parts) < 3:
            continue

        video_id = parts[0]
        language = ".".join(parts[1:-1]).strip()
        if not video_id or not language:
            continue

        if _is_target_subtitle_lang(language):
            subtitle_ids.add(video_id)

    return subtitle_ids


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


def _ensure_subtitle_column(df: DataFrame) -> None:
    if "subtitle_downloaded" not in df.columns:
        df["subtitle_downloaded"] = False


def scan_and_update_progress(
    download_dir: Path = DOWNLOAD_DIR,
    subtitle_dir: Path = SUBTITLE_DIR,
    csv_path: Path = CSV_FILE,
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
    print(f"Scanning subtitles in '{subtitle_dir}'...")
    if subtitle_dir.is_dir():
        subtitle_ids = _load_subtitle_ids(subtitle_dir)
        print(
            "Found "
            + f"{len(subtitle_ids)} videos with yue-Hant/yue subtitle files "
            + "across all subdirectories."
        )
    else:
        subtitle_ids = set()
        print(
            f"Warning: Subtitle directory '{subtitle_dir}' not found. "
            + "Subtitle scan will be treated as empty."
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
    _ensure_subtitle_column(df)

    audio_updates = 0
    subtitle_true_updates = 0
    subtitle_false_updates = 0
    missing_audio_count = 0
    missing_subtitle_count = 0

    records = df.to_dict(orient="records")
    indices = cast(list[Hashable], list(df.index))

    for index_label, record in zip(indices, records):
        video_id = get_video_id(cast(object, record.get("url")))
        if not video_id:
            continue

        is_audio_marked = _is_marked_downloaded(cast(object, record.get("downloaded")))
        is_subtitle_marked = _is_marked_downloaded(
            cast(object, record.get("subtitle_downloaded"))
        )

        if is_audio_marked and (video_id not in downloaded_ids):
            missing_audio_count += 1
        elif not is_audio_marked and (video_id in downloaded_ids):
            df.loc[index_label, "downloaded"] = True
            audio_updates += 1

        if is_subtitle_marked and (video_id not in subtitle_ids):
            missing_subtitle_count += 1
            df.loc[index_label, "subtitle_downloaded"] = False
            subtitle_false_updates += 1
        elif not is_subtitle_marked and (video_id in subtitle_ids):
            df.loc[index_label, "subtitle_downloaded"] = True
            subtitle_true_updates += 1

    total_updates = audio_updates + subtitle_true_updates + subtitle_false_updates
    if total_updates:
        print(
            "Updating CSV rows "
            + f"(downloaded=True: {audio_updates}, "
            + f"subtitle_downloaded=True: {subtitle_true_updates}, "
            + f"subtitle_downloaded=False: {subtitle_false_updates})."
        )
        try:
            df.to_csv(csv_path, index=False)
            print(f"Successfully updated '{csv_path}'.")
        except (OSError, IOError) as exc:
            print(f"Error saving updated CSV file: {exc}")
    else:
        print(
            "No updates needed. CSV file is already in sync with "
            + "download and subtitle folders."
        )

    if missing_audio_count:
        warning_message = (
            f"Warning: Found {missing_audio_count} files marked as 'True' in the CSV "
            + "but not found in the download folder."
        )
        print(warning_message)

    if missing_subtitle_count:
        warning_message = (
            "Warning: Found "
            + f"{missing_subtitle_count} rows marked 'subtitle_downloaded=True' in the CSV "
            + "but no yue-Hant/yue subtitle file in the subtitle folder."
        )
        print(warning_message)


if __name__ == "__main__":
    scan_and_update_progress()
