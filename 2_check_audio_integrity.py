#!/usr/bin/env python3
"""
Check the integrity of downloaded audio files and optionally clean up failures.
"""

from __future__ import annotations

# pyright: reportMissingTypeStubs=false

import argparse
import subprocess
import sys
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas import DataFrame

try:  # pragma: no cover - optional dependency for progress bars
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# --- CONFIGURATION ---
CSV_FILE = Path("legco.csv")
DOWNLOAD_DIR = Path("download")
# --- END CONFIGURATION ---

Status = Literal["valid", "corrupted", "truncated", "suspicious"]


@dataclass(slots=True)
class FileCheckResult:
    """Outcome of validating a single audio file."""

    index: int
    path: Path
    status: Status
    actual_duration: float
    expected_duration: float | None
    detail: float | str | None


@dataclass(slots=True)
class CliArgs:
    """Parsed command-line arguments with concrete types."""

    download_dir: Path
    csv_file: Path
    cleanup: bool
    auto_yes: bool
    summary_only: bool
    processes: int


def get_video_id(url: object) -> str | None:
    """Extract the YouTube video identifier from ``url``."""

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


def parse_duration_string(duration_str: object) -> int | None:
    """Convert HH:MM:SS duration strings into seconds."""

    if not isinstance(duration_str, str):
        return None

    parts = duration_str.strip().split(":")
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


def load_expected_durations(csv_file: Path | str = CSV_FILE) -> dict[str, int]:
    """
    Load expected durations from the metadata CSV.

    Returns:
        Mapping of video ID → expected duration in seconds.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Warning: CSV file {csv_file} not found.")
        return {}
    except (pd.errors.EmptyDataError, OSError) as exc:
        print(f"Warning: Could not load CSV file {csv_file}: {exc}")
        return {}

    expected: dict[str, int] = {}
    records = df.to_dict(orient="records")

    for record in records:
        video_id = get_video_id(cast(object, record.get("url")))
        duration = parse_duration_string(cast(object, record.get("duration")))
        if video_id and duration is not None:
            expected[video_id] = duration

    return expected


def check_opus_file(filepath: Path) -> tuple[bool, float, str | None]:
    """
    Validate that ``filepath`` contains a playable Opus stream.

    Returns:
        ``(is_valid, duration_seconds, error_detail)``.
    """

    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(filepath),
    ]

    try:
        probe_result = subprocess.run(
            probe_cmd, check=True, capture_output=True, text=True
        )
        duration = float(probe_result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        stderr_output = cast(object, exc.stderr)
        if isinstance(stderr_output, str):
            detail = stderr_output.strip()
        elif stderr_output is None:
            detail = ""
        else:
            detail = str(stderr_output).strip()
        return False, 0.0, f"ffprobe error: {detail}"
    except ValueError:
        return False, 0.0, "Cannot parse duration"
    except (OSError, IOError) as exc:
        return False, 0.0, f"Unexpected error: {exc}"

    if duration < 1.0:
        return False, duration, "Duration too short (< 1 second)"

    metadata_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,bit_rate",
        "-of",
        "json",
        str(filepath),
    ]

    metadata_result = subprocess.run(
        metadata_cmd, check=False, capture_output=True, text=True
    )

    if metadata_result.returncode != 0:
        return False, duration, "Cannot read format information"

    return True, duration, None


def format_duration(seconds: float) -> str:
    """Return a human readable version of ``seconds``."""

    if seconds <= 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _status_symbol(status: Status) -> str:
    return {
        "valid": "✓",
        "corrupted": "✗",
        "truncated": "⚠",
        "suspicious": "⚠",
    }.get(status, "?")


def check_file_worker(
    task: tuple[int, Path], expected_durations: dict[str, int]
) -> FileCheckResult:
    """Inspect ``task`` and return a structured result."""

    index, filepath = task
    video_id = filepath.stem

    is_valid, actual_duration, error_detail = check_opus_file(filepath)
    expected_duration = expected_durations.get(video_id)

    if not is_valid:
        return FileCheckResult(
            index=index,
            path=filepath,
            status="corrupted",
            actual_duration=actual_duration,
            expected_duration=expected_duration,
            detail=error_detail,
        )

    if expected_duration is not None:
        diff = abs(actual_duration - expected_duration)
        if diff > 1.0:
            return FileCheckResult(
                index=index,
                path=filepath,
                status="truncated",
                actual_duration=actual_duration,
                expected_duration=expected_duration,
                detail=diff,
            )
        return FileCheckResult(
            index=index,
            path=filepath,
            status="valid",
            actual_duration=actual_duration,
            expected_duration=expected_duration,
            detail=None,
        )

    if actual_duration < 60.0:
        return FileCheckResult(
            index=index,
            path=filepath,
            status="suspicious",
            actual_duration=actual_duration,
            expected_duration=None,
            detail=None,
        )

    return FileCheckResult(
        index=index,
        path=filepath,
        status="valid",
        actual_duration=actual_duration,
        expected_duration=None,
        detail=None,
    )


def _gather_opus_files(directory: Path) -> list[Path]:
    """Return every ``.opus`` file under ``directory``."""

    return sorted(path for path in directory.rglob("*.opus") if path.is_file())


def _gather_m4a_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.m4a") if path.is_file())


def _summarise_results(
    results: list[FileCheckResult],
) -> tuple[
    list[tuple[Path, str]],
    list[tuple[Path, float, float, float]],
    list[tuple[Path, float]],
    int,
    float,
]:
    corrupted: list[tuple[Path, str]] = []
    truncated: list[tuple[Path, float, float, float]] = []
    suspicious: list[tuple[Path, float]] = []
    valid_count = 0
    total_duration = 0.0

    for result in results:
        if result.status == "corrupted":
            corrupted.append((result.path, str(result.detail)))
        elif result.status == "truncated":
            truncated.append(
                (
                    result.path,
                    result.actual_duration,
                    result.expected_duration or 0.0,
                    float(result.detail or 0.0),
                )
            )
        elif result.status == "suspicious":
            suspicious.append((result.path, result.actual_duration))
        else:
            valid_count += 1
            total_duration += result.actual_duration

    return corrupted, truncated, suspicious, valid_count, total_duration


def check_download_directory(
    download_dir: Path = DOWNLOAD_DIR,
    csv_file: Path | str = CSV_FILE,
    *,
    summary_only: bool = False,
    processes: int = 8,
) -> tuple[list[Path], set[str]]:
    """
    Inspect all downloaded Opus files and report issues.

    Returns:
        ``(problematic_paths, problematic_video_ids)``
    """

    if not download_dir.exists():
        print(f"Error: Directory '{download_dir}' not found.")
        return [], set()

    print(f"Loading expected durations from '{csv_file}'...")
    expected_durations = load_expected_durations(csv_file)
    print(f"Loaded expected durations for {len(expected_durations)} videos.\n")

    print(f"Checking audio files in '{download_dir}'...\n")

    opus_files = _gather_opus_files(download_dir)
    if not opus_files:
        print("No opus files found.")
        return [], set()

    proc_label = "process" if processes == 1 else "processes"
    status_message = (
        f"Found {len(opus_files)} opus files. Checking integrity using "
        + f"{processes} parallel {proc_label}...\n"
    )
    print(status_message)

    files_with_indices = list(enumerate(opus_files, start=1))
    check_func = partial(check_file_worker, expected_durations=expected_durations)

    results: list[FileCheckResult] = []

    print("Processing files...")
    with Pool(processes=processes) as pool:
        iterator = pool.imap_unordered(check_func, files_with_indices)
        if tqdm is not None:
            with tqdm(
                total=len(files_with_indices),
                desc="Checking files",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as progress:
                for result in iterator:
                    results.append(result)
                    progress.set_postfix_str(
                        f"{_status_symbol(result.status)} {result.path.stem[:30]}",
                        refresh=False,
                    )
                    _ = progress.update(1)
        else:
            for count, result in enumerate(iterator, start=1):
                results.append(result)
                if count % 100 == 0:
                    print(f"  Processed {count}/{len(files_with_indices)} files...")

    results.sort(key=lambda item: item.index)

    problem_count = sum(1 for item in results if item.status != "valid")
    if tqdm is not None:
        print(f"\n✓ Checked {len(results)} files. Found {problem_count} problematic files.")

    show_details = not summary_only
    detail_lines: list[str] = []

    corrupted, truncated, suspicious, valid_count, total_duration = _summarise_results(
        results
    )

    if show_details:
        for result in results:
            if result.status == "valid":
                continue

            stem = result.path.stem
            prefix = f"[{result.index:4d}/{len(opus_files)}] {stem:.<50}"

            if result.status == "corrupted":
                detail_lines.append(
                    f"{prefix} ✗ CORRUPTED - {result.detail or 'unknown error'}"
                )
            elif result.status == "truncated":
                diff_seconds = float(result.detail or 0.0)
                expected = result.expected_duration or 0.0
                diff_percent = (diff_seconds / expected) * 100 if expected else None
                diff_text = f"{diff_seconds:.1f}s off"
                if diff_percent is not None:
                    diff_text += f", {diff_percent:.1f}% off"
                truncated_line = (
                    f"{prefix} ⚠  TRUNCATED - Expected {format_duration(expected)}, "
                    + f"got {format_duration(result.actual_duration)} ({diff_text})"
                )
                detail_lines.append(truncated_line)
            else:  # suspicious
                detail_lines.append(
                    f"{prefix} ⚠  SUSPICIOUS - Very short ({format_duration(result.actual_duration)})"
                )

    if show_details and detail_lines:
        print("\nDetailed Results:")
        print("-" * 80)
        for line in detail_lines:
            print(line)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total files checked: {len(opus_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Corrupted files: {len(corrupted)}")
    print(f"Truncated files (duration mismatch): {len(truncated)}")
    print(f"Suspicious files (very short): {len(suspicious)}")

    if valid_count:
        print(f"Total duration of valid files: {format_duration(total_duration)}")
        print(f"Average duration: {format_duration(total_duration / valid_count)}")

    if corrupted:
        print("\n✗ CORRUPTED FILES:")
        for path, error in corrupted:
            print(f"  - {path}")
            print(f"    Error: {error}")

    if truncated:
        print("\n⚠  TRUNCATED FILES (significant duration mismatch - likely interrupted):")
        for path, actual, expected, diff_seconds in truncated:
            diff_percent = (diff_seconds / expected) * 100 if expected else None
            diff_text = f"{diff_seconds:.1f}s off"
            if diff_percent is not None:
                diff_text += f", {diff_percent:.1f}% off"
            truncated_message = (
                f"  - {path}\n"
                + f"    Expected: {format_duration(expected)}, "
                + f"got {format_duration(actual)} ({diff_text})"
            )
            print(truncated_message)

    if suspicious:
        print("\n⚠  SUSPICIOUS FILES (very short, no expected duration):")
        for path, duration in suspicious:
            print(f"  - {path} (duration: {format_duration(duration)})")

    print("\n" + "=" * 60)
    print("Checking for unconverted m4a files...")
    m4a_files = _gather_m4a_files(download_dir)
    if m4a_files:
        print(f"\n⚠  Found {len(m4a_files)} unconverted m4a files:")
        for path in m4a_files:
            print(f"  - {path}")
        print("These files may indicate interrupted conversions.")

    corrupted_paths = [path for path, _ in corrupted]
    truncated_paths = [path for path, _, _, _ in truncated]
    suspicious_paths = [path for path, _ in suspicious]

    all_problematic = corrupted_paths + truncated_paths + suspicious_paths
    video_ids = {path.stem for path in all_problematic}

    if all_problematic and show_details:
        print("\n" + "=" * 60)
        print("ALL PROBLEMATIC OPUS FILES (for easy copying/processing):")
        print("-" * 60)
        for path in sorted(all_problematic):
            print(path)
        print("-" * 60)
        print(f"Total problematic files: {len(all_problematic)}")

        print("\n" + "=" * 60)
        print("VIDEO IDs TO RE-DOWNLOAD:")
        print("-" * 60)
        for video_id in sorted(video_ids):
            print(video_id)
        print("-" * 60)
        print(f"Total videos to re-download: {len(video_ids)}")

    return all_problematic, video_ids


def _delete_files(paths: Iterable[Path]) -> int:
    count = 0
    for path in paths:
        try:
            path.unlink()
            print(f"✓ Deleted: {path}")
            count += 1
        except (OSError, IOError) as exc:
            print(f"✗ Failed to delete {path}: {exc}")
    return count


def _downloaded_count(df: DataFrame) -> int:
    if "downloaded" not in df.columns:
        return 0

    total = cast(object, df["downloaded"].sum())
    if isinstance(total, (int, float, bool)):
        return int(total)
    return 0


def _update_csv_flags(csv_path: Path | str, video_ids: set[str]) -> tuple[int, int]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"✗ Failed to update CSV: {csv_path} not found")
        return 0, 0
    except (pd.errors.EmptyDataError, OSError) as exc:
        print(f"✗ Failed to update CSV: {exc}")
        return 0, 0

    updated = 0
    original_count = _downloaded_count(df)

    records = df.to_dict(orient="records")
    indices = cast(list[Hashable], list(df.index))

    for index_label, record in zip(indices, records):
        if get_video_id(cast(object, record.get("url"))) in video_ids:
            df.loc[index_label, "downloaded"] = False
            updated += 1

    df.to_csv(csv_path, index=False)
    new_count = _downloaded_count(df)
    return updated, int(original_count - new_count)


def perform_cleanup(
    problematic_files: list[Path],
    video_ids: set[str],
    csv_path: Path | str,
    *,
    require_confirmation: bool,
) -> None:
    if not problematic_files:
        print("\n✓ No problematic files found. Nothing to clean up!")
        return

    print("\n" + "=" * 60)
    print("CLEANUP OPERATION")
    print("=" * 60)

    if require_confirmation:
        print("\nThis will:")
        print(f"1. Delete {len(problematic_files)} problematic opus files")
        print(f"2. Mark {len(video_ids)} videos as not downloaded in {csv_path}")

        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response not in {"yes", "y"}:
            print("Operation cancelled.")
            return
    else:
        print("\nRunning in auto-yes mode; skipping confirmation prompt.")

    print("\nDeleting files...")
    deleted = _delete_files(problematic_files)
    print(f"\nDeleted {deleted}/{len(problematic_files)} files.")

    print(f"\nUpdating {csv_path}...")
    updated_rows, difference = _update_csv_flags(csv_path, video_ids)
    if updated_rows:
        print("✓ Updated CSV successfully")
        print(f"  - Marked {updated_rows} videos as not downloaded")
        print(f"  - Downloaded count delta: {-difference}")
    else:
        print("No CSV rows required changes.")


def main() -> None:
    try:
        _ = subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffprobe not found. Please install ffmpeg.")
        print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Check integrity of downloaded audio files and optionally clean up problematic files."
    )
    _ = parser.add_argument(
        "download_dir",
        nargs="?",
        default=str(DOWNLOAD_DIR),
        help="Directory containing downloaded files (default: download/)",
    )
    _ = parser.add_argument(
        "csv_file",
        nargs="?",
        default=str(CSV_FILE),
        help="CSV file with video metadata (default: legco.csv)",
    )
    _ = parser.add_argument(
        "--cleanup", action="store_true", help="Delete problematic files and update CSV"
    )
    _ = parser.add_argument(
        "--auto-yes",
        action="store_true",
        help="Automatically answer yes to cleanup confirmation",
    )
    _ = parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary without detailed file-by-file results",
    )
    _ = parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of parallel processes to use (default: 8)",
    )

    args_namespace = parser.parse_args()

    download_dir_raw = cast(str, args_namespace.download_dir)
    csv_file_raw = cast(str, args_namespace.csv_file)
    cleanup_flag = cast(bool, args_namespace.cleanup)
    auto_yes_flag = cast(bool, args_namespace.auto_yes)
    summary_only_flag = cast(bool, args_namespace.summary_only)
    processes_value = cast(int, args_namespace.processes)

    download_dir_value = Path(download_dir_raw)
    csv_file_value = Path(csv_file_raw)

    cli_args = CliArgs(
        download_dir=download_dir_value,
        csv_file=csv_file_value,
        cleanup=cleanup_flag,
        auto_yes=auto_yes_flag,
        summary_only=summary_only_flag,
        processes=processes_value,
    )

    if cli_args.processes < 1:
        parser.error("--processes must be at least 1")

    download_dir = cli_args.download_dir
    csv_path = cli_args.csv_file

    print(f"Using download directory: {download_dir}")
    print(f"Using CSV file: {csv_path}\n")

    problematic_files, video_ids = check_download_directory(
        download_dir=download_dir,
        csv_file=csv_path,
        summary_only=cli_args.summary_only,
        processes=cli_args.processes,
    )

    if cli_args.cleanup:
        perform_cleanup(
            problematic_files,
            video_ids,
            csv_path,
            require_confirmation=not cli_args.auto_yes,
        )


if __name__ == "__main__":
    main()
