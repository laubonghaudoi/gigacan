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
TRANSCRIPTIONS_DIR = Path("transcriptions")
YUE_SUBTITLE_DIR = Path("yue")
ZH_HK_SUBTITLE_DIR = Path("zh-hk")
SUBTITLE_TARGET_CONFIG = {
    "subtitle_downloaded": {
        "langs": {"yue-hant", "yue"},
        "roots": (YUE_SUBTITLE_DIR,),
    },
    "zh-hk_downloaded": {
        "langs": {"zh-hk"},
        "roots": (ZH_HK_SUBTITLE_DIR,),
    },
}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ttml", ".lrc", ".json3"}
TRANSCRIBED_COLUMN = "transcribed"
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


def _load_transcribed_ids(transcriptions_dir: Path) -> set[str]:
    """Return the set of video IDs inferred from ``transcriptions_dir`` SRT files."""

    transcribed_ids: set[str] = set()
    if not transcriptions_dir.is_dir():
        return transcribed_ids

    for path in transcriptions_dir.rglob("*.srt"):
        if path.is_file():
            transcribed_ids.add(path.stem)
    return transcribed_ids


def _normalise_lang(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _is_target_subtitle_lang(lang: str, target_langs: set[str]) -> bool:
    lang_norm = _normalise_lang(lang)
    if lang_norm in target_langs:
        return True
    if any(
        lang_norm.startswith(target + "-") for target in sorted(target_langs)
    ):
        return True
    return False


def _extract_video_id_and_lang(path: Path) -> tuple[str, str] | None:
    parts = path.name.split(".")
    if len(parts) < 3:
        return None

    video_id = parts[0]
    language = ".".join(parts[1:-1]).strip()
    if not video_id or not language:
        return None
    return video_id, language


def _load_subtitle_ids_by_column() -> dict[str, set[str]]:
    """Return video IDs for each subtitle target column across configured roots."""

    subtitle_ids_by_column = {column: set() for column in SUBTITLE_TARGET_CONFIG}
    for column, config in SUBTITLE_TARGET_CONFIG.items():
        target_langs = cast(set[str], config["langs"])
        roots = cast(tuple[Path, ...], config["roots"])
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
                    continue
                parsed = _extract_video_id_and_lang(path)
                if parsed is None:
                    continue
                video_id, language = parsed
                if _is_target_subtitle_lang(language, target_langs):
                    subtitle_ids_by_column[column].add(video_id)

    return subtitle_ids_by_column


def _is_marked_downloaded(value: object) -> bool:
    """Interpret the ``downloaded`` column values as booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _ensure_download_column(df: DataFrame) -> bool:
    changed = False
    if "downloaded" not in df.columns:
        df["downloaded"] = False
        changed = True
    return changed


def _ensure_subtitle_columns(df: DataFrame) -> bool:
    changed = False
    for column in SUBTITLE_TARGET_CONFIG:
        if column not in df.columns:
            df[column] = False
            changed = True
    return changed


def _ensure_transcribed_column(df: DataFrame) -> bool:
    if TRANSCRIBED_COLUMN not in df.columns:
        df[TRANSCRIBED_COLUMN] = False
        return True
    return False


def scan_and_update_progress(
    download_dir: Path = DOWNLOAD_DIR,
    csv_path: Path = CSV_FILE,
    transcriptions_dir: Path = TRANSCRIPTIONS_DIR,
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
    print(f"Scanning for transcriptions in '{transcriptions_dir}'...")
    transcribed_ids = _load_transcribed_ids(transcriptions_dir)
    print(
        f"Found {len(transcribed_ids)} transcribed SRT files across all subdirectories."
    )
    print(
        "Scanning subtitles in configured roots: "
        + f"yue='{YUE_SUBTITLE_DIR}', zh-hk='{ZH_HK_SUBTITLE_DIR}'..."
    )
    subtitle_ids_by_column = _load_subtitle_ids_by_column()
    print(
        "Found "
        + f"{len(subtitle_ids_by_column['subtitle_downloaded'])} videos with yue-Hant/yue subtitle files "
        + f"and {len(subtitle_ids_by_column['zh-hk_downloaded'])} videos with zh-HK subtitle files "
        + "across configured roots."
    )

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return
    except (pd.errors.EmptyDataError, OSError) as exc:
        print(f"Error reading CSV file '{csv_path}': {exc}")
        return

    schema_changed = False
    schema_changed = _ensure_download_column(df) or schema_changed
    schema_changed = _ensure_subtitle_columns(df) or schema_changed
    schema_changed = _ensure_transcribed_column(df) or schema_changed

    audio_updates = 0
    transcribed_true_updates = 0
    transcribed_false_updates = 0
    subtitle_true_updates = {column: 0 for column in SUBTITLE_TARGET_CONFIG}
    subtitle_false_updates = {column: 0 for column in SUBTITLE_TARGET_CONFIG}
    missing_audio_count = 0
    missing_transcribed_count = 0
    missing_subtitle_count = {column: 0 for column in SUBTITLE_TARGET_CONFIG}

    records = df.to_dict(orient="records")
    indices = cast(list[Hashable], list(df.index))

    for index_label, record in zip(indices, records):
        video_id = get_video_id(cast(object, record.get("url")))
        if not video_id:
            continue

        is_audio_marked = _is_marked_downloaded(cast(object, record.get("downloaded")))
        if is_audio_marked and (video_id not in downloaded_ids):
            missing_audio_count += 1
        elif not is_audio_marked and (video_id in downloaded_ids):
            df.loc[index_label, "downloaded"] = True
            audio_updates += 1

        is_transcribed_marked = _is_marked_downloaded(
            cast(object, record.get(TRANSCRIBED_COLUMN))
        )
        if is_transcribed_marked and (video_id not in transcribed_ids):
            missing_transcribed_count += 1
            df.loc[index_label, TRANSCRIBED_COLUMN] = False
            transcribed_false_updates += 1
        elif not is_transcribed_marked and (video_id in transcribed_ids):
            df.loc[index_label, TRANSCRIBED_COLUMN] = True
            transcribed_true_updates += 1

        for column in SUBTITLE_TARGET_CONFIG:
            is_subtitle_marked = _is_marked_downloaded(cast(object, record.get(column)))
            subtitle_ids = subtitle_ids_by_column[column]
            if is_subtitle_marked and (video_id not in subtitle_ids):
                missing_subtitle_count[column] += 1
                df.loc[index_label, column] = False
                subtitle_false_updates[column] += 1
            elif not is_subtitle_marked and (video_id in subtitle_ids):
                df.loc[index_label, column] = True
                subtitle_true_updates[column] += 1

    total_updates = (
        audio_updates
        + transcribed_true_updates
        + transcribed_false_updates
        + sum(subtitle_true_updates.values())
        + sum(subtitle_false_updates.values())
    )
    if total_updates or schema_changed:
        print(
            "Updating CSV rows "
            + f"(downloaded=True: {audio_updates}, "
            + f"transcribed=True: {transcribed_true_updates}, "
            + f"transcribed=False: {transcribed_false_updates}, "
            + f"subtitle_downloaded=True: {subtitle_true_updates['subtitle_downloaded']}, "
            + f"subtitle_downloaded=False: {subtitle_false_updates['subtitle_downloaded']}, "
            + f"zh-hk_downloaded=True: {subtitle_true_updates['zh-hk_downloaded']}, "
            + f"zh-hk_downloaded=False: {subtitle_false_updates['zh-hk_downloaded']})."
        )
        try:
            df.to_csv(csv_path, index=False)
            print(f"Successfully updated '{csv_path}'.")
        except (OSError, IOError) as exc:
            print(f"Error saving updated CSV file: {exc}")
    else:
        print(
            "No updates needed. CSV file is already in sync with "
            + "download, transcriptions, and subtitle folders."
        )

    if missing_audio_count:
        warning_message = (
            f"Warning: Found {missing_audio_count} files marked as 'True' in the CSV "
            + "but not found in the download folder."
        )
        print(warning_message)

    if missing_subtitle_count["subtitle_downloaded"]:
        warning_message = (
            "Warning: Found "
            + f"{missing_subtitle_count['subtitle_downloaded']} rows marked 'subtitle_downloaded=True' in the CSV "
            + "but no yue-Hant/yue subtitle file in configured subtitle roots."
        )
        print(warning_message)

    if missing_transcribed_count:
        warning_message = (
            f"Warning: Found {missing_transcribed_count} rows marked '{TRANSCRIBED_COLUMN}=True' in the CSV "
            + "but no SRT file in the transcriptions folder."
        )
        print(warning_message)

    if missing_subtitle_count["zh-hk_downloaded"]:
        warning_message = (
            "Warning: Found "
            + f"{missing_subtitle_count['zh-hk_downloaded']} rows marked 'zh-hk_downloaded=True' in the CSV "
            + "but no zh-HK subtitle file in configured subtitle roots."
        )
        print(warning_message)


if __name__ == "__main__":
    scan_and_update_progress()
