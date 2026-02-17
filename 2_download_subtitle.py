from __future__ import annotations

# pyright: reportMissingTypeStubs=false

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas import DataFrame
from yt_dlp import YoutubeDL


# --- CONFIGURATION ---
CSV_FILE = Path("legco.csv")
SUBTITLE_DIR = Path("subtitle")
COOKIES_FILE = Path("cookies.txt")
TARGET_LANG_PRIORITY = ["yue-Hant", "yue"]
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ttml", ".lrc", ".json3"}
# --- END CONFIGURATION ---


@dataclass(slots=True)
class SubtitleTask:
    """Work item describing one video subtitle download target."""

    index: Hashable
    url: str
    video_id: str
    year: str


@dataclass(slots=True)
class SubtitleSelection:
    """Chosen subtitle source and language."""

    source: str  # "manual" or "auto"
    language: str


def get_video_id(url: object) -> str | None:
    """Return the YouTube video ID embedded in ``url``."""

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


def _normalise_lang(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _normalise_year(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and pd.isna(value):
        return "unknown"

    text = str(value).strip()
    if not text:
        return "unknown"

    year = text[:4]
    if year.isdigit():
        return year
    return "unknown"


def _is_marked_downloaded(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return False
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _ensure_subtitle_downloaded_column(df: DataFrame) -> None:
    if "subtitle_downloaded" not in df.columns:
        df["subtitle_downloaded"] = False
    else:
        df["subtitle_downloaded"] = df["subtitle_downloaded"].map(_is_marked_downloaded)


def _normalise_publish_date(df: DataFrame) -> None:
    if "publish_date" not in df.columns:
        return

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df.sort_values(by="publish_date", ascending=False, inplace=True, ignore_index=False)


def _base_ydl_options() -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if COOKIES_FILE.exists():
        options["cookiefile"] = str(COOKIES_FILE)
    return options


def _extract_video_info(url: str) -> dict[str, Any] | None:
    with YoutubeDL(_base_ydl_options()) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        return None

    if info.get("_type") == "playlist":
        entries = info.get("entries")
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    return cast(dict[str, Any], item)
        return None

    return cast(dict[str, Any], info)


def _normalise_subtitles(raw: object) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {}

    subtitles: dict[str, list[dict[str, Any]]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, list):
            continue
        entries = [item for item in value if isinstance(item, dict)]
        if entries:
            subtitles[key] = entries
    return subtitles


def _choose_matching_lang(candidates: list[str], target: str) -> str | None:
    if not candidates:
        return None

    target_norm = _normalise_lang(target)
    exact = [lang for lang in candidates if _normalise_lang(lang) == target_norm]
    if exact:
        return sorted(exact)[0]

    prefixed = [
        lang
        for lang in candidates
        if _normalise_lang(lang).startswith(target_norm + "-")
    ]
    if prefixed:
        return sorted(prefixed)[0]

    return None


def select_subtitle_track(info: dict[str, Any]) -> SubtitleSelection | None:
    """Pick yue-Hant first, then yue; manual preferred over auto for same language."""

    manual = _normalise_subtitles(info.get("subtitles"))
    auto = _normalise_subtitles(info.get("automatic_captions"))

    manual_keys = [key for key in manual if _normalise_lang(key) != "live-chat"]
    auto_keys = [key for key in auto if _normalise_lang(key) != "live-chat"]

    for target in TARGET_LANG_PRIORITY:
        manual_lang = _choose_matching_lang(manual_keys, target)
        if manual_lang:
            return SubtitleSelection(source="manual", language=manual_lang)

        auto_lang = _choose_matching_lang(auto_keys, target)
        if auto_lang:
            return SubtitleSelection(source="auto", language=auto_lang)

    return None


def _subtitle_dir_for_year(year: str) -> Path:
    return SUBTITLE_DIR / year


def _subtitle_suffix_priority(path: Path) -> tuple[int, str]:
    suffix = path.suffix.lower()
    return (0 if suffix == ".srt" else 1, path.name)


def _extract_lang_from_filename(path: Path, video_id: str) -> str:
    name = path.name
    prefix = f"{video_id}."
    if not name.startswith(prefix):
        return ""
    stem_part = name[len(prefix) :]
    if "." not in stem_part:
        return ""
    return stem_part.rsplit(".", 1)[0]


def find_existing_target_subtitle(video_id: str, year: str) -> Path | None:
    year_dir = _subtitle_dir_for_year(year)
    if not year_dir.is_dir():
        return None

    candidates: list[Path] = []
    for path in year_dir.glob(f"{video_id}.*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue
        lang = _extract_lang_from_filename(path, video_id)
        lang_norm = _normalise_lang(lang)
        if lang_norm == "yue-hant" or lang_norm == "yue":
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates, key=_subtitle_suffix_priority)[0]


def download_subtitle(task: SubtitleTask, selection: SubtitleSelection) -> Path | None:
    year_dir = _subtitle_dir_for_year(task.year)
    year_dir.mkdir(parents=True, exist_ok=True)

    options = _base_ydl_options()
    options.update(
        {
            "outtmpl": str(year_dir / "%(id)s.%(ext)s"),
            "subtitleslangs": [selection.language],
            "subtitlesformat": "srt/best",
            "convertsubtitles": "srt",
        }
    )
    if selection.source == "manual":
        options["writesubtitles"] = True
    else:
        options["writeautomaticsub"] = True

    with YoutubeDL(options) as ydl:
        code = ydl.download([task.url])
    if code != 0:
        return None

    exact_srt = year_dir / f"{task.video_id}.{selection.language}.srt"
    if exact_srt.exists():
        return exact_srt

    exact_any = sorted(
        year_dir.glob(f"{task.video_id}.{selection.language}.*"),
        key=_subtitle_suffix_priority,
    )
    if exact_any:
        return exact_any[0]

    fallback = find_existing_target_subtitle(task.video_id, task.year)
    return fallback


def load_tasks(df: DataFrame) -> list[SubtitleTask]:
    tasks: list[SubtitleTask] = []
    records = cast(dict[Hashable, dict[str, object]], df.to_dict(orient="index"))

    for index, record in records.items():
        url_value = record.get("url")
        if not isinstance(url_value, str):
            continue

        video_id = get_video_id(url_value)
        if not video_id:
            print(f"Could not extract video ID from URL: {url_value}")
            continue

        year = _normalise_year(record.get("publish_date"))
        marked = _is_marked_downloaded(record.get("subtitle_downloaded"))
        existing = find_existing_target_subtitle(video_id, year)

        if marked and existing is not None:
            continue

        tasks.append(
            SubtitleTask(
                index=index,
                url=url_value,
                video_id=video_id,
                year=year,
            )
        )

    return tasks


def process_task(df: DataFrame, task: SubtitleTask) -> bool:
    existing = find_existing_target_subtitle(task.video_id, task.year)
    if existing is not None:
        df.loc[task.index, "subtitle_downloaded"] = True
        print(f"[skip] {task.video_id}: already has yue subtitle at {existing}")
        return True

    try:
        info = _extract_video_info(task.url)
    except Exception as exc:
        print(f"[error] {task.video_id}: failed to read metadata ({exc})")
        df.loc[task.index, "subtitle_downloaded"] = False
        return False

    if info is None:
        print(f"[skip] {task.video_id}: could not fetch metadata")
        df.loc[task.index, "subtitle_downloaded"] = False
        return False

    selection = select_subtitle_track(info)
    if selection is None:
        print(f"[skip] {task.video_id}: no yue-Hant/yue subtitle available")
        df.loc[task.index, "subtitle_downloaded"] = False
        return True

    try:
        subtitle_path = download_subtitle(task, selection)
    except Exception as exc:
        print(f"[error] {task.video_id}: subtitle download failed ({exc})")
        df.loc[task.index, "subtitle_downloaded"] = False
        return False

    if subtitle_path is None:
        print(
            f"[skip] {task.video_id}: selected {selection.source}/{selection.language} "
            + "but no subtitle file was produced"
        )
        df.loc[task.index, "subtitle_downloaded"] = False
        return False

    df.loc[task.index, "subtitle_downloaded"] = True
    print(
        f"[ok] {task.video_id}: downloaded {selection.source}/{selection.language} -> "
        + f"{subtitle_path}"
    )
    return True


def main() -> None:
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    _ensure_subtitle_downloaded_column(df)
    _normalise_publish_date(df)

    if not COOKIES_FILE.exists():
        print(
            "Warning: cookies.txt not found; subtitle metadata/download may fail with HTTP 403."
        )

    tasks = load_tasks(df)
    if not tasks:
        print("No pending subtitle tasks.")
        df.to_csv(CSV_FILE, index=False)
        return

    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] Processing {task.video_id} ({task.year}) ...")
        if process_task(df, task):
            completed += 1

    df.to_csv(CSV_FILE, index=False)
    print(
        "Done. Processed "
        + f"{len(tasks)} videos; successful/handled: {completed}; "
        + f"CSV updated: {CSV_FILE}"
    )


if __name__ == "__main__":
    main()
