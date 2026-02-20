#!/usr/bin/env python3
"""Probe local audio durations with ffprobe and update legco.csv."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


CSV_FILE = Path("legco.csv")
DOWNLOAD_DIR = Path("download")
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


@dataclass(slots=True)
class CliArgs:
    csv_file: Path
    download_dir: Path
    workers: int
    overwrite: bool
    no_backup: bool


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Probe audio durations from local files with ffprobe and update "
            "the duration column in legco.csv."
        )
    )
    parser.add_argument("--csv", type=Path, default=CSV_FILE, help=f"CSV path (default: {CSV_FILE})")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help=f"Audio root directory (default: {DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "CPU worker count. 0 = auto (2x logical CPU count to maximize throughput). "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-probe rows that already have a duration value.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak backup before overwriting CSV.",
    )
    ns = parser.parse_args()
    if ns.workers < 0:
        parser.error("--workers must be >= 0")
    return CliArgs(
        csv_file=ns.csv,
        download_dir=ns.download_dir,
        workers=ns.workers,
        overwrite=ns.overwrite,
        no_backup=ns.no_backup,
    )


def resolve_worker_count(workers: int) -> int:
    if workers > 0:
        return workers
    cpu_count = max(1, os.cpu_count() or 1)
    return max(1, cpu_count * 2)


def get_video_id(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    if "youtu.be" in url:
        parts = url.rstrip("/").split("/")
        if not parts:
            return None
        candidate = parts[-1].split("?")[0].strip()
        return candidate or None

    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"youtube.com", "www.youtube.com"}:
        return None
    candidate = (parse_qs(parsed.query).get("v") or [""])[0].strip()
    return candidate or None


def parse_duration_seconds(raw: object) -> int | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
            total = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = (int(part) for part in parts)
            total = minutes * 60 + seconds
        else:
            total = int(float(value))
    except ValueError:
        return None
    if total <= 0:
        return None
    return total


def format_hms(seconds: int) -> str:
    safe = max(0, int(seconds))
    hours, rem = divmod(safe, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_audio_index(download_dir: Path) -> dict[str, Path]:
    if not download_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {download_dir}")
    mapping: dict[str, Path] = {}
    for path in sorted(download_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        mapping.setdefault(path.stem, path)
    return mapping


def probe_duration_seconds_ffprobe(path_str: str) -> int | None:
    audio_path = Path(path_str)
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install FFmpeg and retry.") from exc
    except subprocess.CalledProcessError:
        return None

    try:
        seconds = float(output)
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return int(round(seconds))


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()

    if not args.csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

    print(f"Indexing audio files under {args.download_dir}...")
    audio_by_id = build_audio_index(args.download_dir)
    print(f"Indexed {len(audio_by_id)} audio files.")

    with args.csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = [dict(row) for row in reader]

    if "duration" not in fieldnames:
        fieldnames.append("duration")
        for row in rows:
            row.setdefault("duration", "")

    has_duration_seconds = "duration_seconds" in fieldnames
    tasks: list[tuple[int, str, Path]] = []
    missing_id = 0
    missing_audio = 0
    skipped_existing = 0

    for idx, row in enumerate(rows):
        video_id = get_video_id(row.get("url"))
        if not video_id:
            missing_id += 1
            continue

        audio_path = audio_by_id.get(video_id)
        if audio_path is None:
            missing_audio += 1
            continue

        existing = parse_duration_seconds(row.get("duration"))
        if existing is None and has_duration_seconds:
            existing = parse_duration_seconds(row.get("duration_seconds"))
        if existing is not None and not args.overwrite:
            skipped_existing += 1
            continue

        tasks.append((idx, video_id, audio_path))

    print(
        "Probe plan: "
        f"rows={len(rows)}, "
        f"to_probe={len(tasks)}, "
        f"skipped_existing={skipped_existing}, "
        f"missing_id={missing_id}, "
        f"missing_audio={missing_audio}"
    )

    if not tasks:
        print("No rows require probing; CSV unchanged.")
        return

    workers = resolve_worker_count(args.workers)
    print(f"Using workers={workers}")

    completed = 0
    failed = 0
    progress = (
        tqdm(total=len(tasks), desc="Probing durations", unit="file")
        if tqdm is not None and sys.stdout.isatty()
        else None
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(probe_duration_seconds_ffprobe, str(audio_path)): (idx, video_id)
            for idx, video_id, audio_path in tasks
        }
        for future in as_completed(futures):
            idx, _video_id = futures[future]
            try:
                seconds = future.result()
            except RuntimeError as exc:
                if progress is not None:
                    progress.close()
                raise SystemExit(str(exc)) from exc
            except Exception:
                failed += 1
                if progress is not None:
                    progress.update(1)
                continue

            if seconds is None:
                failed += 1
                if progress is not None:
                    progress.update(1)
                continue

            rows[idx]["duration"] = format_hms(seconds)
            if has_duration_seconds:
                rows[idx]["duration_seconds"] = str(seconds)
            completed += 1
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    if not args.no_backup:
        backup_path = args.csv_file.with_suffix(args.csv_file.suffix + ".bak")
        backup_path.write_bytes(args.csv_file.read_bytes())
        print(f"Wrote backup: {backup_path}")

    write_csv_atomic(args.csv_file, fieldnames, rows)
    print(
        "Done: "
        f"updated={completed}, "
        f"failed={failed}, "
        f"output={args.csv_file}"
    )


if __name__ == "__main__":
    main()
