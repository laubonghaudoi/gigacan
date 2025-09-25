#!/usr/bin/env python3
"""Bundle .opus audio and metadata into WebDataset shards."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(slots=True)
class Sample:
    row: dict[str, str]
    audio_path: Path
    year: str
    size_bytes: int
    mtime: int
    sort_key: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata_csv",
        type=Path,
        help="Path to CSV containing audio metadata.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for WebDataset shards.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help=(
            "Base directory for audio files if paths in the CSV are relative."
            " Defaults to metadata CSV parent."
        ),
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=20,
        help="Maximum number of samples per shard (default: 20).",
    )
    parser.add_argument(
        "--max-shard-size-mb",
        type=float,
        default=1500.0,
        help="Maximum shard size in megabytes before rolling over (default: 1500).",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        help="Optional list of years to include (e.g. 2015 2016).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned shards without writing tar files.",
    )
    parser.add_argument(
        "--jobs",
        type=_parse_jobs,
        default=8,
        help=(
            "Number of parallel workers to use when writing shards. "
            "Use 0 to autodetect (default: autodetect)."
        ),
    )
    return parser.parse_args()


def _parse_jobs(value: str) -> int:
    jobs = int(value)
    if jobs < 0:
        raise argparse.ArgumentTypeError("jobs must be >= 0")
    return jobs


def load_samples(args: argparse.Namespace) -> dict[str, list[Sample]]:
    metadata_path = args.metadata_csv
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    audio_root = args.audio_root or metadata_path.parent
    year_filter = set(args.years) if args.years else None

    samples_by_year: dict[str, list[Sample]] = {}

    with metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            audio_rel = (row.get("audio") or "").strip()
            if not audio_rel:
                warn(f"Skipping row without audio path: {row}")
                continue

            audio_path = (audio_root / audio_rel).resolve() if not Path(audio_rel).is_absolute() else Path(audio_rel)
            if not audio_path.is_file():
                warn(f"Audio file not found: {audio_path}")
                continue

            year = infer_year(row, audio_path)
            if year_filter and year not in year_filter:
                continue

            stat_result = audio_path.stat()
            size_bytes = stat_result.st_size
            mtime = int(stat_result.st_mtime)
            sort_key = build_sort_key(row, audio_rel)
            sample = Sample(
                row=row,
                audio_path=audio_path,
                year=year,
                size_bytes=size_bytes,
                mtime=mtime,
                sort_key=sort_key,
            )
            samples_by_year.setdefault(year, []).append(sample)

    for year_samples in samples_by_year.values():
        year_samples.sort(key=lambda s: s.sort_key)

    return samples_by_year


def infer_year(row: dict[str, str], audio_path: Path) -> str:
    publish_date = (row.get("publish_date") or "").strip()
    if len(publish_date) >= 4 and publish_date[:4].isdigit():
        return publish_date[:4]

    for part in audio_path.parts:
        if len(part) == 4 and part.isdigit():
            return part

    raise ValueError(f"Unable to infer year for audio: {audio_path}")


def build_sort_key(row: dict[str, str], audio_rel: str) -> tuple[str, ...]:
    publish_date = (row.get("publish_date") or "").strip()
    title = (row.get("title") or "").strip()
    identifier = (row.get("id") or "").strip()
    return (publish_date, title, identifier, audio_rel)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{int(num_bytes)} B"


def write_shards(
    samples_by_year: dict[str, list[Sample]],
    output_dir: Path,
    samples_per_shard: int,
    max_shard_size_mb: float,
    dry_run: bool,
    jobs: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_shard_size_mb * 1024 * 1024)
    shard_tasks: list[tuple[str, Path, int, tuple[Sample, ...], int, bool]] = []

    for year, samples in sorted(samples_by_year.items()):
        if not samples:
            continue

        shard_dir = output_dir / year
        shard_dir.mkdir(parents=True, exist_ok=True)

        batch: list[Sample] = []
        batch_size = 0
        shard_index = 0
        global_index = 0

        for sample in samples:
            exceeds_samples = len(batch) >= samples_per_shard
            exceeds_size = batch and (batch_size + sample.size_bytes > max_bytes)
            if batch and (exceeds_samples or exceeds_size):
                shard_tasks.append(
                    (
                        year,
                        shard_dir,
                        shard_index,
                        tuple(batch),
                        global_index - len(batch),
                        dry_run,
                    )
                )
                shard_index += 1
                batch = []
                batch_size = 0

            batch.append(sample)
            batch_size += sample.size_bytes
            global_index += 1

        if batch:
            shard_tasks.append(
                (
                    year,
                    shard_dir,
                    shard_index,
                    tuple(batch),
                    global_index - len(batch),
                    dry_run,
                )
            )

    if not shard_tasks:
        summary_prefix = "Planned" if dry_run else "Created"
        print(f"{summary_prefix} 0 shard(s) in total.")
        return

    worker_count = determine_worker_count(jobs)

    if dry_run or worker_count == 1:
        for task in shard_tasks:
            message = process_shard(*task)
            print(message)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_shard, *task) for task in shard_tasks]
            for future in futures:
                print(future.result())

    summary_prefix = "Planned" if dry_run else "Created"
    print(f"{summary_prefix} {len(shard_tasks)} shard(s) in total.")


def determine_worker_count(jobs: int | None) -> int:
    if jobs is None or jobs == 0:
        return max(1, os.cpu_count() or 1)
    return jobs


def process_shard(
    year: str,
    shard_dir: Path,
    shard_index: int,
    samples: tuple[Sample, ...],
    base_index: int,
    dry_run: bool,
) -> str:
    shard_name = f"legco-{year}-{shard_index:05d}.tar"
    shard_path = shard_dir / shard_name

    entries: list[tuple[Path, str, str, bytes, int]] = []
    total_bytes = 0

    for offset, sample in enumerate(samples):
        sample_index = base_index + offset
        base_filename = f"legco-{year}-{sample_index:06d}"
        audio_member = f"{base_filename}.opus"
        json_member = f"{base_filename}.json"

        payload = dict(sample.row)
        payload["audio"] = audio_member
        payload_bytes = json.dumps(payload, ensure_ascii=False, indent=None).encode("utf-8")
        mtime = sample.mtime

        entries.append((sample.audio_path, audio_member, json_member, payload_bytes, mtime))
        total_bytes += sample.size_bytes + len(payload_bytes)

    human_size = format_size(total_bytes)

    if dry_run:
        return f"[DRY-RUN] {year} → {shard_name}: {len(samples)} samples (~{human_size})"

    with tarfile.open(shard_path, "w") as tar:
        for audio_path, audio_member, json_member, payload_bytes, mtime in entries:
            tar.add(audio_path, arcname=audio_member)

            info = tarfile.TarInfo(name=json_member)
            info.size = len(payload_bytes)
            info.mtime = mtime
            tar.addfile(info, io.BytesIO(payload_bytes))

    return f"Wrote {shard_path} with {len(samples)} samples (~{human_size})"


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    samples_by_year = load_samples(args)
    if not samples_by_year:
        print("No samples matched the provided filters.", file=sys.stderr)
        return

    write_shards(
        samples_by_year,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        max_shard_size_mb=args.max_shard_size_mb,
        dry_run=args.dry_run,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
