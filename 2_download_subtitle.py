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
COOKIES_FILE = Path("cookies.txt")
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ttml", ".lrc", ".json3"}
# --- END CONFIGURATION ---


@dataclass(frozen=True, slots=True)
class SubtitleTarget:
    """One subtitle tracking target in the CSV."""

    column: str
    language_priority: tuple[str, ...]
    label: str
    output_root: Path


TARGETS: tuple[SubtitleTarget, ...] = (
    SubtitleTarget(
        column="subtitle_downloaded",
        language_priority=("yue-Hant", "yue"),
        label="yue-Hant/yue",
        output_root=Path("yue"),
    ),
    SubtitleTarget(
        column="zh-hk_downloaded",
        language_priority=("zh-HK",),
        label="zh-HK",
        output_root=Path("zh-hk"),
    ),
)


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


def _ensure_target_columns(df: DataFrame) -> None:
    for target in TARGETS:
        if target.column not in df.columns:
            df[target.column] = False
        else:
            df[target.column] = df[target.column].map(_is_marked_downloaded)


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


def _lang_matches_target(candidate: str, target: str) -> bool:
    candidate_norm = _normalise_lang(candidate)
    target_norm = _normalise_lang(target)
    return candidate_norm == target_norm or candidate_norm.startswith(target_norm + "-")


def _lang_matches_any_target(candidate: str, targets: tuple[str, ...]) -> bool:
    return any(_lang_matches_target(candidate, target) for target in targets)


def _choose_matching_lang(candidates: list[str], target: str) -> str | None:
    if not candidates:
        return None

    exact = [lang for lang in candidates if _lang_matches_target(lang, target)]
    if exact:
        return sorted(exact)[0]

    return None


def select_subtitle_track(
    info: dict[str, Any],
    language_priority: tuple[str, ...],
) -> SubtitleSelection | None:
    """Select manual first, then auto, according to ``language_priority``."""

    manual = _normalise_subtitles(info.get("subtitles"))
    auto = _normalise_subtitles(info.get("automatic_captions"))

    manual_keys = [key for key in manual if _normalise_lang(key) != "live-chat"]
    auto_keys = [key for key in auto if _normalise_lang(key) != "live-chat"]

    for target in language_priority:
        manual_lang = _choose_matching_lang(manual_keys, target)
        if manual_lang:
            return SubtitleSelection(source="manual", language=manual_lang)

        auto_lang = _choose_matching_lang(auto_keys, target)
        if auto_lang:
            return SubtitleSelection(source="auto", language=auto_lang)

    return None


def _subtitle_dir_for_year(target: SubtitleTarget, year: str) -> Path:
    return target.output_root / year


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


def find_existing_target_subtitle(
    video_id: str,
    year: str,
    target: SubtitleTarget,
) -> Path | None:
    year_dir = _subtitle_dir_for_year(target, year)
    if not year_dir.is_dir():
        return None

    candidates: list[Path] = []
    for path in year_dir.glob(f"{video_id}.*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue
        lang = _extract_lang_from_filename(path, video_id)
        if not lang:
            continue
        if _lang_matches_any_target(lang, target.language_priority):
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates, key=_subtitle_suffix_priority)[0]


def download_subtitle(
    task: SubtitleTask,
    target: SubtitleTarget,
    selection: SubtitleSelection,
) -> Path | None:
    year_dir = _subtitle_dir_for_year(target, task.year)
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

    return find_existing_target_subtitle(
        task.video_id,
        task.year,
        target,
    )


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

        done_for_all_targets = True
        for target in TARGETS:
            marked = _is_marked_downloaded(record.get(target.column))
            if not marked:
                done_for_all_targets = False
                break

        if done_for_all_targets:
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
    row_had_error = False

    marked_map: dict[str, bool] = {}
    for target in TARGETS:
        marked_map[target.column] = _is_marked_downloaded(df.loc[task.index, target.column])

    pending_targets: list[SubtitleTarget] = []
    for target in TARGETS:
        if marked_map[target.column]:
            print(
                f"[skip] {task.video_id}: {target.label} marked downloaded in CSV ({target.column}=True)"
            )
            continue

        existing = find_existing_target_subtitle(task.video_id, task.year, target)
        if existing is not None:
            df.loc[task.index, target.column] = True
            print(
                f"[skip] {task.video_id}: already has {target.label} subtitle at {existing}"
            )
            continue

        pending_targets.append(target)

    if not pending_targets:
        return True

    try:
        info = _extract_video_info(task.url)
    except Exception as exc:
        print(f"[error] {task.video_id}: failed to read metadata ({exc})")
        for target in TARGETS:
            if not marked_map[target.column]:
                df.loc[task.index, target.column] = False
        return False

    if info is None:
        print(f"[skip] {task.video_id}: could not fetch metadata")
        for target in TARGETS:
            if not marked_map[target.column]:
                df.loc[task.index, target.column] = False
        return False

    for target in pending_targets:
        selection = select_subtitle_track(info, target.language_priority)
        if selection is None:
            print(
                f"[skip] {task.video_id}: no {target.label} subtitle available"
            )
            df.loc[task.index, target.column] = False
            continue

        try:
            subtitle_path = download_subtitle(task, target, selection)
        except Exception as exc:
            print(
                f"[error] {task.video_id}: {target.label} subtitle download failed ({exc})"
            )
            df.loc[task.index, target.column] = False
            row_had_error = True
            continue

        if subtitle_path is None:
            print(
                f"[skip] {task.video_id}: selected {target.label} "
                + f"({selection.source}/{selection.language}) but no file was produced"
            )
            df.loc[task.index, target.column] = False
            row_had_error = True
            continue

        df.loc[task.index, target.column] = True
        print(
            f"[ok] {task.video_id}: downloaded {target.label} "
            + f"({selection.source}/{selection.language}) -> {subtitle_path}"
        )

    return not row_had_error


def main() -> None:
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    _ensure_target_columns(df)
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
