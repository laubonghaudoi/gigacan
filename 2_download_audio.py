from __future__ import annotations

# pyright: reportMissingTypeStubs=false

import subprocess
from collections.abc import Hashable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas import DataFrame


# --- CONFIGURATION ---
CSV_FILE = Path("legco.csv")
DOWNLOAD_DIR = Path("download")
COOKIES_FILE = Path("cookies.txt")
AUDIO_FORMAT = "opus"
TARGET_SAMPLE_RATE = "16000"
POOL_PROCESSES = 8
# --- END CONFIGURATION ---


@dataclass(slots=True)
class DownloadTask:
    """Work item describing a video that still needs audio downloaded."""

    index: Hashable
    url: str
    video_id: str


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


def build_download_command(output_path: Path, url: str) -> list[str]:
    """Compose the ``yt-dlp`` invocation for ``url``."""

    command: list[str] = [
        "yt-dlp",
        "-o",
        str(output_path),
        "-x",
        "--audio-format",
        AUDIO_FORMAT,
        "--postprocessor-args",
        f"-ar {TARGET_SAMPLE_RATE}",
    ]

    if COOKIES_FILE.exists():
        command.extend(["--cookies", str(COOKIES_FILE)])

    command.append(url)
    return command


def download_audio(task: DownloadTask) -> bool:
    """Download and convert the audio track for ``task``."""

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DOWNLOAD_DIR / f"{task.video_id}.opus"
    command = build_download_command(output_path, task.url)

    try:
        _ = subprocess.run(command, check=True, capture_output=False)
    except subprocess.CalledProcessError as exc:
        print(f"An error occurred while downloading {task.url}: {exc}")
        return False
    except FileNotFoundError:
        print("yt-dlp not found. Please ensure it is installed and in your PATH.")
        return False

    print(f"Audio downloaded and converted successfully: {output_path}")
    return True


def download_worker(task: DownloadTask) -> Hashable | None:
    """Multiprocessing entry point – returns the DataFrame index on success."""

    if download_audio(task):
        return task.index

    return None


def load_pending_tasks(df: DataFrame) -> list[DownloadTask]:
    """Extract download tasks from the CSV-backed DataFrame."""

    tasks: list[DownloadTask] = []
    records = cast(dict[Hashable, dict[str, object]], df.to_dict(orient="index"))

    for index, record in records.items():
        url_value = record.get("url")
        downloaded_value = record.get("downloaded")

        if not isinstance(url_value, str):
            continue

        video_id = get_video_id(url_value)
        if not video_id:
            print(f"Could not extract video ID from URL: {url_value}")
            continue

        if _is_marked_downloaded(downloaded_value):
            continue

        tasks.append(DownloadTask(index=index, url=url_value, video_id=video_id))

    return tasks


def _is_marked_downloaded(value: object) -> bool:
    """Return ``True`` if a CSV cell denotes that the video was already downloaded."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return False
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _ensure_download_column(df: DataFrame) -> None:
    """Guarantee that the ``downloaded`` column exists."""

    if "downloaded" not in df.columns:
        df["downloaded"] = False


def _normalise_publish_date(df: DataFrame) -> None:
    """Coerce publish dates to datetimes so we can sort newest first."""

    if "publish_date" not in df.columns:
        return

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df.sort_values(by="publish_date", ascending=False, inplace=True, ignore_index=False)


def main() -> None:
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    _ensure_download_column(df)
    _normalise_publish_date(df)

    tasks = load_pending_tasks(df)
    if not tasks:
        print("All videos have already been processed.")
        return

    if not COOKIES_FILE.exists():
        warning_message = (
            "Warning: cookies.txt not found; downloads may fail with HTTP 403. "
            + "Export fresh cookies if requests start failing."
        )
        print(warning_message)

    print(f"Starting download of {len(tasks)} videos using {POOL_PROCESSES} processes...")

    with Pool(processes=POOL_PROCESSES) as pool:
        results = pool.map(download_worker, tasks)

    successful_indices = [index for index in results if index is not None]

    for index in successful_indices:
        df.loc[index, "downloaded"] = True

    if successful_indices:
        df.to_csv(CSV_FILE, index=False)
        print(f"Updated {CSV_FILE} with {len(successful_indices)} new downloads.")

    print("All processing is complete.")


if __name__ == "__main__":
    main()
