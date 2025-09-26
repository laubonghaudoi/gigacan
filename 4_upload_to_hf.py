import argparse
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
from huggingface_hub import HfApi, HfFolder, hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError

# --- CONFIGURATION ---
CSV_FILE = Path('metadata.csv')
AUDIO_ROOT = Path('download')
DATASET_DESCRIPTION = (
    'Raw Legislative Council session audio captured as `.opus` files, '
    'grouped by publication year and paired with basic metadata for discovery.'
)


DEFAULT_CONFIG_NAME = 'raw'
# --- END CONFIGURATION ---


def sanitize_text(value: object) -> str:
    """Return a safe string for metadata fields, collapsing missing values."""
    if pd.isna(value):
        return ''
    return str(value)

def parse_duration_seconds(duration: object) -> int:
    """Convert `HH:MM:SS` strings to total seconds. Returns 0 on failure."""
    if pd.isna(duration):
        return 0
    text = str(duration)
    parts = text.split(':')
    if len(parts) != 3:
        return 0
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return 0
    return hours * 3600 + minutes * 60 + seconds

def upload_year_to_hf(
    year: int,
    repo_id: str,
    config_name: str = DEFAULT_CONFIG_NAME,
    metadata_csv: Path = CSV_FILE,
    audio_root: Path = AUDIO_ROOT,
    staging_dir: Path | None = None,
    revision: str = "main",
) -> None:
    """Upload a single year's audio and metadata to the raw Hugging Face config."""
    print(f"--- Starting upload process for year {year} ---")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)

    def delete_remote_file(path_in_repo: str) -> None:
        try:
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type='dataset',
                revision=revision,
            )
            print(f'Removed obsolete file from Hub: {path_in_repo}')
        except EntryNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - cleanup best effort
            print(f'Warning: could not remove {path_in_repo}: {exc}')

    metadata_csv = Path(metadata_csv)
    audio_root = Path(audio_root)
    metadata_dir = metadata_csv.parent.resolve()
    audio_root = audio_root.resolve()

    if not metadata_csv.is_file():
        print(f"Error: metadata CSV not found at {metadata_csv}.")
        return

    try:
        main_df = pd.read_csv(metadata_csv)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error: failed to read {metadata_csv}: {exc}")
        return

    if 'publish_date' not in main_df.columns:
        print('Error: metadata CSV is missing a "publish_date" column.')
        return
    path_columns = [column for column in ('audio', 'file_path') if column in main_df.columns]
    if not path_columns:
        print('Error: metadata CSV is missing an audio path column (expected "audio" or "file_path").')
        return
    source_path_col = 'audio' if 'audio' in path_columns else path_columns[0]

    year_str = str(year)
    year_df = main_df[main_df['publish_date'].astype(str).str.startswith(year_str, na=False)].copy()
    if year_df.empty:
        print(f'No rows found for year {year_str} in {metadata_csv}.')
        return

    def resolve_audio_path(cell) -> Path | None:
        if pd.isna(cell):
            return None
        path_str = str(cell).strip()
        if not path_str:
            return None
        raw_path = Path(path_str)

        candidate_paths: List[Path] = []

        def add_candidate(candidate: Path) -> None:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:  # pragma: no cover - best effort resolution
                resolved = candidate
            if resolved not in candidate_paths:
                candidate_paths.append(resolved)

        if raw_path.is_absolute():
            add_candidate(raw_path)
        else:
            add_candidate(metadata_dir / raw_path)
            add_candidate(audio_root / raw_path)
            parts = raw_path.parts
            if parts and parts[0] == config_name:
                remainder_parts = parts[1:]
                if remainder_parts:
                    add_candidate(audio_root.joinpath(*remainder_parts))

        for candidate in candidate_paths:
            if candidate.is_file():
                return candidate
        return None

    year_df['audio_path'] = year_df[source_path_col].apply(resolve_audio_path)
    missing_audio = year_df[year_df['audio_path'].isna()]
    if not missing_audio.empty:
        print(
            f"Warning: {len(missing_audio)} metadata rows are missing audio files on disk; "
            "they will be skipped."
        )
    year_df = year_df[year_df['audio_path'].notna()].copy()
    if year_df.empty:
        print('No matching audio files found for the selected year.')
        return

    year_df['audio_path'] = year_df['audio_path'].apply(lambda path: path.resolve())
    year_df['file_name'] = year_df['audio_path'].apply(lambda path: path.name)

    parent_dirs = {path.parent for path in year_df['audio_path']}
    if len(parent_dirs) != 1:
        locations = ', '.join(str(path) for path in sorted(parent_dirs))
        print(
            'Error: audio files span multiple directories '
            f'({locations}). Consolidate them before uploading.'
        )
        return
    local_year_dir = parent_dirs.pop()
    expected_year_dir = (audio_root / year_str).resolve()
    if expected_year_dir != local_year_dir:
        print(
            f"Info: resolved audio lives in {local_year_dir}, which differs from "
            f"the configured root {expected_year_dir}. Proceeding with the resolved path."
        )

    available_on_disk = {path.name for path in local_year_dir.glob('*.opus')}
    expected_files = set(year_df['file_name'])
    missing_on_disk = expected_files - available_on_disk
    if missing_on_disk:
        print(
            f"Warning: {len(missing_on_disk)} expected audio files are missing in {local_year_dir}."
        )
    unused_audio = available_on_disk - expected_files
    if unused_audio:
        print(f"Skipping {len(unused_audio)} .opus files without metadata entries.")

    year_df = year_df[year_df['file_name'].isin(available_on_disk)].copy()
    if year_df.empty:
        print('No metadata rows map to existing audio files after filtering.')
        return

    def extract_duration_seconds(row) -> int:
        raw_value = getattr(row, 'duration_seconds', None)
        if raw_value is not None and not pd.isna(raw_value):
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                pass
        return parse_duration_seconds(getattr(row, 'duration', None))

    metadata_rows: List[dict] = []
    total_audio_bytes = 0

    for row in year_df.itertuples(index=False):
        audio_path: Path = getattr(row, 'audio_path')
        file_name: str = getattr(row, 'file_name')
        repo_audio_path = f'{config_name}/{year_str}/{file_name}'

        total_audio_bytes += audio_path.stat().st_size
        duration_seconds = extract_duration_seconds(row)

        metadata_row = {
            'id': sanitize_text(getattr(row, 'id', file_name)),
            'file_path': repo_audio_path,
            'title': sanitize_text(getattr(row, 'title', '')),
            'description': sanitize_text(getattr(row, 'description', '')),
            'publish_date': sanitize_text(getattr(row, 'publish_date', '')),
            'duration': sanitize_text(getattr(row, 'duration', '')),
            'duration_seconds': int(duration_seconds),
        }
        if 'url' in year_df.columns:
            metadata_row['url'] = sanitize_text(getattr(row, 'url', ''))
        metadata_rows.append(metadata_row)

    if not metadata_rows:
        print('No metadata rows could be prepared for upload.')
        return

    temp_parent = staging_dir if staging_dir is not None else local_year_dir.parent
    temp_parent = Path(temp_parent)
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(temp_parent)) as staging_root_str:
        staging_root = Path(staging_root_str)
        target_dir = staging_root / config_name / year_str
        target_dir.mkdir(parents=True, exist_ok=True)

        linked_files = 0
        copied_files = 0
        for row in year_df.itertuples(index=False):
            audio_path: Path = getattr(row, 'audio_path')
            destination = target_dir / getattr(row, 'file_name')
            try:
                os.link(audio_path, destination)
                linked_files += 1
            except OSError:
                shutil.copy2(audio_path, destination)
                copied_files += 1

        if copied_files:
            print(
                f"Note: created {copied_files} temporary copies (hard linking failed for some files)."
            )
        if linked_files:
            print(f"Linked {linked_files} files into staging directory.")

        print(
            f"Uploading {len(metadata_rows)} clips (~{total_audio_bytes / (1024 ** 3):.2f} GB)"
            f" for year {year_str} (config: {config_name})..."
        )
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type='dataset',
            folder_path=str(staging_root),
            revision=revision,
        )
        print('Audio upload complete.')

    delete_remote_file(f"{config_name}/{year_str}/metadata.jsonl")
    delete_remote_file('dataset_infos.json')

    metadata_filename = 'metadata.csv'
    new_metadata_df = pd.DataFrame(metadata_rows)
    ordered_columns = [
        column
        for column in [
            'id',
            'file_path',
            'title',
            'description',
            'publish_date',
            'duration',
            'duration_seconds',
            'url',
        ]
        if column in new_metadata_df.columns
    ]
    new_metadata_df = new_metadata_df[ordered_columns]

    for column in ['file_path', 'title', 'description', 'publish_date', 'duration', 'url']:
        if column in new_metadata_df.columns:
            new_metadata_df[column] = new_metadata_df[column].apply(sanitize_text)
    if 'duration_seconds' in new_metadata_df.columns:
        new_metadata_df['duration_seconds'] = (
            new_metadata_df['duration_seconds'].fillna(0).astype(int)
        )

    try:
        existing_metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename=metadata_filename,
            repo_type='dataset',
            revision=revision,
        )
        old_metadata_df = pd.read_csv(existing_metadata_path)
        if 'audio' in old_metadata_df.columns and 'file_path' not in old_metadata_df.columns:
            old_metadata_df = old_metadata_df.rename(columns={'audio': 'file_path'})
        print('Found existing metadata. Merging new data.')
    except Exception:
        old_metadata_df = pd.DataFrame(columns=new_metadata_df.columns)
        print('No existing metadata found. Creating a new one.')

    combined_df = pd.concat(
        [old_metadata_df, new_metadata_df], ignore_index=True, sort=False
    ).drop_duplicates(subset='id', keep='last')

    if 'publish_date' in combined_df.columns:
        combined_df['publish_date'] = combined_df['publish_date'].fillna('').astype(str)
    for column in ['file_path', 'title', 'description', 'duration', 'url']:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].fillna('').astype(str)
    if 'duration_seconds' not in combined_df.columns:
        combined_df['duration_seconds'] = 0
    combined_df['duration_seconds'] = combined_df['duration_seconds'].fillna(0).astype(int)
    if 'audio' in combined_df.columns:
        combined_df = combined_df.drop(columns=['audio'])

    sort_columns = [
        column for column in ['publish_date', 'duration_seconds'] if column in combined_df.columns
    ]
    if sort_columns:
        combined_df = combined_df.sort_values(
            by=sort_columns,
            ascending=[False] * len(sort_columns),
            na_position='last',
        )

    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    upload_file(
        path_or_fileobj=csv_buffer.getvalue().encode('utf-8'),
        path_in_repo=metadata_filename,
        repo_id=repo_id,
        repo_type='dataset',
        revision=revision,
    )
    print('Metadata CSV upload complete.')

    print(f"\n--- Process for year {year} complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a year of audio data to the Hugging Face Hub."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="The publication year to upload (e.g. 2025).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="laubonghaudoi/legco",
        help="The Hugging Face repository ID to upload to.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help=(
            "Top-level directory name to store audio within the dataset repo (e.g. 'raw')."
            " The same value is used when writing relative paths into metadata.csv."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=CSV_FILE,
        help="Path to the metadata CSV generated by the pipeline.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=AUDIO_ROOT,
        help="Root directory containing per-year audio subfolders.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to host the temporary upload workspace (defaults to the"
            " audio root's parent). Point this to a fast local disk when working from"
            " network storage."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repository revision/branch to target (default: main).",
    )

    args = parser.parse_args()

    if not HfFolder.get_token():
        print(
            "Hugging Face token not found. Please log in first using 'huggingface-cli login'."
        )
    else:
        upload_year_to_hf(
            year=args.year,
            repo_id=args.repo_id,
            config_name=args.config_name,
            metadata_csv=args.metadata_csv,
            audio_root=args.audio_root,
            staging_dir=args.staging_dir,
            revision=args.revision,
        )
