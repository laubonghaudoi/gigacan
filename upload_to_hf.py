import os
import json
import argparse
import io
import tempfile
import shutil
from typing import Dict, List

import pandas as pd
import yaml
from datasets import DatasetInfo, Features, Value, Audio
from datasets.splits import SplitDict, SplitInfo
from datasets.utils.version import Version
from datasets.utils.py_utils import asdict
from huggingface_hub import HfApi, HfFolder, hf_hub_download, upload_file
from urllib.parse import urlparse, parse_qs

# --- CONFIGURATION ---
CSV_FILE = 'legco_20250920.csv'
DOWNLOAD_DIR = 'download/'
RAW_CONFIG_NAME = 'raw'
DATASET_VERSION = Version('1.0.0')
DATASET_DESCRIPTION = (
    'Raw Legislative Council session audio captured as `.opus` files, '
    'grouped by publication year and paired with basic metadata for discovery.'
)
DEFAULT_SAMPLING_RATE = 16000


DATASET_FEATURES = Features({
    'id': Value('string'),
    'audio': Audio(sampling_rate=DEFAULT_SAMPLING_RATE, decode=True),
    'title': Value('string'),
    'description': Value('string'),
    'publish_date': Value('string'),
    'duration': Value('string'),
    'duration_seconds': Value('int64'),
})
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



def load_dataset_infos(repo_id: str) -> Dict[str, dict]:
    """Fetch existing dataset configuration information from the Hub if present."""
    try:
        dataset_info_path = hf_hub_download(
            repo_id=repo_id, filename='dataset_infos.json', repo_type='dataset'
        )
    except Exception:
        return {}

    with open(dataset_info_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        return {}

    return payload


def save_dataset_infos(repo_id: str, dataset_infos: Dict[str, dict]) -> None:
    """Persist dataset configuration back to the Hub."""
    buffer = io.BytesIO()
    buffer.write(json.dumps(dataset_infos, ensure_ascii=False, indent=2).encode('utf-8'))
    buffer.seek(0)
    upload_file(
        path_or_fileobj=buffer,
        path_in_repo='dataset_infos.json',
        repo_id=repo_id,
        repo_type='dataset',
    )


def update_raw_dataset_info(
    repo_id: str,
    dataset_infos: Dict[str, dict],
    year: str,
    metadata_rel_path: str,
    num_examples: int,
    split_dataset_size: int,
) -> None:
    """Ensure the raw config advertises the provided year split."""
    existing_raw_info = dataset_infos.get(RAW_CONFIG_NAME, {})
    existing_data_files = existing_raw_info.get('data_files', {})

    raw_info_obj = DatasetInfo.from_dict(existing_raw_info)
    raw_info_obj.description = DATASET_DESCRIPTION
    raw_info_obj.features = DATASET_FEATURES
    raw_info_obj.builder_name = raw_info_obj.builder_name or 'json'
    raw_info_obj.config_name = RAW_CONFIG_NAME
    raw_info_obj.version = raw_info_obj.version or DATASET_VERSION

    splits = raw_info_obj.splits if raw_info_obj.splits is not None else SplitDict()
    splits[year] = SplitInfo(
        name=year,
        num_bytes=int(split_dataset_size),
        num_examples=int(num_examples),
        shard_lengths=[int(num_examples)] if num_examples else [],
        dataset_name=RAW_CONFIG_NAME,
    )
    raw_info_obj.splits = splits

    total_dataset_size = sum(split.num_bytes or 0 for split in raw_info_obj.splits.values())
    raw_info_obj.dataset_size = total_dataset_size or None
    raw_info_obj.download_size = total_dataset_size or None
    raw_info_obj.size_in_bytes = total_dataset_size or None

    serialized = asdict(raw_info_obj)
    audio_feature = serialized.get('features', {}).get('audio')
    if isinstance(audio_feature, dict):
        audio_feature.setdefault('sampling_rate', DEFAULT_SAMPLING_RATE)
        audio_feature.setdefault('decode', True)

    data_files: Dict[str, List[str]] = {}
    for split_name, files in existing_data_files.items():
        if isinstance(files, (list, tuple)):
            data_files[split_name] = [str(file) for file in files]
        else:
            data_files[split_name] = [str(files)]
    data_files[year] = [metadata_rel_path]
    serialized['data_files'] = data_files

    dataset_infos[RAW_CONFIG_NAME] = serialized
    save_dataset_infos(repo_id, dataset_infos)

def get_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    if pd.isna(url):
        return None
    try:
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            video_id = parse_qs(parsed_url.query).get('v')
            if video_id:
                return video_id[0]
    except Exception as e:
        print(f"Could not parse URL '{url}': {e}")
    return None

def generate_dataset_card(repo_id: str, split_counts: Dict[str, int], total_examples: int) -> str:
    """Generates the content for the dataset card (README.md)."""
    card_data = {
        'configs': [{'config_name': RAW_CONFIG_NAME, 'data_files': []}]
    }
    for year in sorted(split_counts):
        card_data['configs'][0]['data_files'].append({
            'split': year,
            'path': f'{RAW_CONFIG_NAME}/{year}/metadata.jsonl'
        })
    yaml_header = f"---\n{yaml.dump(card_data, default_flow_style=False)}---\n"

    if split_counts:
        table_rows = ['| Year | Clips |', '| --- | --- |']
        for year in sorted(split_counts):
            table_rows.append(f"| {year} | {split_counts[year]} |")
        table_section = '\n'.join(table_rows)
        example_year = next(iter(sorted(split_counts)))
    else:
        table_section = 'No splits available yet.'
        example_year = '2025'

    readme_content = f"""
# Dataset for {repo_id}

This dataset contains {total_examples} raw Legislative Council audio clips organised by publication year.

## Available splits
{table_section}

## Usage
```python
from datasets import load_dataset

year = "{example_year}"
ds = load_dataset("{repo_id}", "{RAW_CONFIG_NAME}", split=year)
sample = ds[0]
print(sample["title"], sample["audio"]["path"])  # local path within the repo
audio = sample["audio"]  # access `path` and waveform data once decoded
```
"""
    return yaml_header + readme_content

def upload_year_to_hf(year: int, repo_id: str) -> None:
    """Upload a single year's audio and metadata to the raw Hugging Face config."""
    print(f"--- Starting upload process for year {year} ---")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)

    year_str = str(year)
    local_year_dir = os.path.join(DOWNLOAD_DIR, year_str)
    if not os.path.isdir(local_year_dir):
        print(f"Error: Directory for year {year_str} not found.")
        return

    main_df = pd.read_csv(CSV_FILE)
    year_df = main_df[main_df['publish_date'].str.startswith(year_str, na=False)].copy()

    year_df['id'] = year_df['url'].apply(get_video_id)
    year_df = year_df.dropna(subset=['id'])
    year_df['file_name'] = year_df['id'].apply(lambda x: f"{x}.opus")

    expected_files = set(year_df['file_name'])
    available_audio = {
        filename: os.path.join(local_year_dir, filename)
        for filename in os.listdir(local_year_dir)
        if filename.endswith('.opus')
    }

    missing_on_disk = expected_files - set(available_audio)
    if missing_on_disk:
        print(
            f"Warning: {len(missing_on_disk)} expected audio files are missing in {local_year_dir}."
        )

    year_df = year_df[year_df['file_name'].isin(available_audio)].copy()
    if year_df.empty:
        print('No matching audio files found for the selected year.')
        return

    unused_audio = set(available_audio) - set(expected_files)
    if unused_audio:
        print(f"Skipping {len(unused_audio)} .opus files without metadata entries.")

    year_df['local_path'] = year_df['file_name'].map(available_audio)
    year_df = year_df.drop_duplicates(subset='id')

    records = []
    for row in year_df.itertuples(index=False):
        records.append({
            'id': sanitize_text(row.id),
            'audio': {'path': f'{RAW_CONFIG_NAME}/{year_str}/{row.file_name}', 'bytes': None},
            'title': sanitize_text(row.title),
            'description': sanitize_text(row.description),
            'publish_date': sanitize_text(row.publish_date),
            'duration': sanitize_text(row.duration),
            'duration_seconds': parse_duration_seconds(row.duration),
        })

    if not records:
        print('No metadata rows could be prepared for upload.')
        return

    print(f"Preparing {len(records)} clips for upload...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_upload_path = os.path.join(temp_dir, RAW_CONFIG_NAME, year_str)
        os.makedirs(temp_upload_path)

        for row in year_df.itertuples(index=False):
            shutil.copy(row.local_path, os.path.join(temp_upload_path, row.file_name))

        metadata_json_path = os.path.join(temp_upload_path, 'metadata.jsonl')
        with open(metadata_json_path, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

        metadata_num_bytes = os.path.getsize(metadata_json_path)
        audio_total_bytes = sum(
            os.path.getsize(os.path.join(temp_upload_path, row.file_name))
            for row in year_df.itertuples(index=False)
        )

        print(f"Starting resumable upload for {len(records)} audio files...")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type='dataset',
            folder_path=temp_dir,
            allow_patterns=[f"{RAW_CONFIG_NAME}/{year_str}/*"],
        )
        print('Audio and metadata upload complete.')

    metadata_rel_path = f"{RAW_CONFIG_NAME}/{year_str}/metadata.jsonl"
    num_examples = len(records)
    split_dataset_size = metadata_num_bytes + audio_total_bytes

    dataset_infos = load_dataset_infos(repo_id)
    update_raw_dataset_info(
        repo_id=repo_id,
        dataset_infos=dataset_infos,
        year=year_str,
        metadata_rel_path=metadata_rel_path,
        num_examples=num_examples,
        split_dataset_size=split_dataset_size,
    )
    print('Dataset configuration updated.')

    metadata_filename = f"{RAW_CONFIG_NAME}/metadata.csv"
    year_df = year_df.drop(columns=['local_path'])
    new_metadata_df = year_df[[
        'id', 'file_name', 'url', 'title', 'description', 'publish_date', 'duration'
    ]].copy()
    new_metadata_df['duration_seconds'] = new_metadata_df['duration'].apply(parse_duration_seconds)
    new_metadata_df['id'] = new_metadata_df['id'].astype(str)
    new_metadata_df['file_name'] = new_metadata_df['file_name'].astype(str)
    for column in ['url', 'title', 'description', 'publish_date', 'duration']:
        new_metadata_df[column] = new_metadata_df[column].apply(sanitize_text)

    try:
        existing_metadata_path = hf_hub_download(
            repo_id=repo_id, filename=metadata_filename, repo_type='dataset'
        )
        old_metadata_df = pd.read_csv(existing_metadata_path)
        print('Found existing metadata. Merging new data.')
    except Exception:
        old_metadata_df = pd.DataFrame(columns=new_metadata_df.columns)
        print('No existing metadata found. Creating a new one.')

    combined_df = (
        pd.concat([old_metadata_df, new_metadata_df], ignore_index=True)
        .drop_duplicates(subset='id', keep='last')
    )
    combined_df = combined_df.sort_values(
        by=['publish_date', 'duration_seconds'],
        ascending=[False, False],
        na_position='last'
    )
    combined_df['publish_date'] = combined_df['publish_date'].fillna('')
    for column in ['url', 'title', 'description', 'duration']:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].fillna('')
    if 'duration_seconds' in combined_df.columns:
        combined_df['duration_seconds'] = combined_df['duration_seconds'].fillna(0).astype(int)

    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    upload_file(
        path_or_fileobj=csv_buffer.getvalue().encode('utf-8'),
        path_in_repo=metadata_filename,
        repo_id=repo_id,
        repo_type='dataset',
    )
    print('Metadata CSV upload complete.')

    publish_years = combined_df['publish_date'].astype(str).str[:4]
    publish_years = publish_years[publish_years.str.isdigit()]
    split_counts = publish_years.value_counts().sort_index().to_dict()
    total_examples = int(len(combined_df))

    readme_content = generate_dataset_card(repo_id, split_counts, total_examples)
    upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='dataset',
    )
    print('Dataset card has been updated.')
    print(f"\n--- Process for year {year} complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a year of audio data to Hugging Face Hub.")
    parser.add_argument("--year", type=int, required=True, help="The year of the data to upload.")
    parser.add_argument("--repo_id", type=str, default="laubonghaudoi/legco", help="The Hugging Face repository ID.")
    
    args = parser.parse_args()
    
    if not HfFolder.get_token():
        print("Hugging Face token not found. Please log in first using 'huggingface-cli login'.")
    else:
        upload_year_to_hf(args.year, args.repo_id)
