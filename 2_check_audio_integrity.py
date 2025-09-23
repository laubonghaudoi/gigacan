#!/usr/bin/env python3
"""
Script to check the integrity of downloaded audio files.
Identifies potentially corrupted or incomplete opus files.
Compares actual durations with expected durations from the CSV file.
"""

import os
import subprocess
import sys
import pandas as pd
from urllib.parse import urlparse, parse_qs
from multiprocessing import Pool
from functools import partial
import argparse
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# --- CONFIGURATION ---
CSV_FILE = 'legco_20250920.csv'
DOWNLOAD_DIR = 'download/'
# --- END CONFIGURATION ---


def get_video_id(url):
    """
    Extracts the YouTube video ID from a URL.
    """
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
    except (ValueError, AttributeError, TypeError):
        pass
    return None


def parse_duration_string(duration_str):
    """
    Convert duration string from HH:MM:SS to seconds.
    Returns None if parsing fails.
    """
    if pd.isna(duration_str):
        return None
    try:
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return int(duration_str)
    except (ValueError, AttributeError):
        return None


def load_expected_durations(csv_file=CSV_FILE):
    """
    Load expected durations from CSV file.
    Returns a dictionary mapping video_id to expected duration in seconds.
    """
    try:
        df = pd.read_csv(csv_file)
        video_durations = {}

        for _, row in df.iterrows():
            video_id = get_video_id(row['url'])
            expected_duration = parse_duration_string(row['duration'])

            if video_id and expected_duration:
                video_durations[video_id] = expected_duration

        return video_durations
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"Warning: Could not load CSV file: {e}")
        return {}


def check_opus_file(filepath):
    """
    Check if an opus file is valid and complete.
    Returns (is_valid, duration, error_message)
    """
    try:
        # Use ffprobe to check the file
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(filepath)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        # Check if duration is reasonable (at least 1 second)
        if duration < 1.0:
            return False, duration, "Duration too short (< 1 second)"

        # Additional check: try to get format info
        cmd_format = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=format_name,bit_rate',
            '-of', 'json',
            str(filepath)
        ]

        result_format = subprocess.run(cmd_format, capture_output=True, text=True, check=False)
        if result_format.returncode != 0:
            return False, duration, "Cannot read format information"

        return True, duration, None

    except subprocess.CalledProcessError as e:
        return False, 0, f"ffprobe error: {e.stderr}"
    except ValueError:
        return False, 0, "Cannot parse duration"
    except (OSError, IOError) as e:
        return False, 0, f"Unexpected error: {str(e)}"


def format_duration(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def check_file_worker(filepath_with_index, expected_durations):
    """
    Worker function to check a single file. 
    Returns (index, filepath, status, actual_duration, expected_duration, error)
    where status is one of: 'valid', 'corrupted', 'truncated', 'suspicious'
    """
    i, filepath = filepath_with_index
    video_id = os.path.splitext(os.path.basename(filepath))[0]

    # Check the file
    is_valid, actual_duration, error = check_opus_file(filepath)
    expected_duration = expected_durations.get(video_id)

    if not is_valid:
        return (i, filepath, 'corrupted', actual_duration, expected_duration, error)
    else:
        # Check duration mismatch if we have expected duration
        if expected_duration:
            duration_diff = abs(actual_duration - expected_duration)
            duration_diff_percent = (duration_diff / expected_duration) * 100

            if duration_diff_percent > 10:  # More than 10% difference
                return (i, filepath, 'truncated', actual_duration, expected_duration, duration_diff_percent)
            else:
                return (i, filepath, 'valid', actual_duration, expected_duration, None)
        elif actual_duration < 60:  # Less than 1 minute is suspicious if no expected duration
            return (i, filepath, 'suspicious', actual_duration, expected_duration, None)
        else:
            return (i, filepath, 'valid', actual_duration, expected_duration, None)


def check_download_directory(download_dir=DOWNLOAD_DIR, csv_file=CSV_FILE, summary_only=False):
    """
    Check all opus files in the download directory and its subdirectories.
    Compare with expected durations from CSV file.
    """
    if not os.path.exists(download_dir):
        print(f"Error: Directory '{download_dir}' not found.")
        return

    print(f"Loading expected durations from '{csv_file}'...")
    expected_durations = load_expected_durations(csv_file)
    print(f"Loaded expected durations for {len(expected_durations)} videos.\n")

    print(f"Checking audio files in '{download_dir}'...\n")

    # Find all opus files
    opus_files = []
    for root, _, files in os.walk(download_dir):
        for file in files:
            if file.endswith('.opus'):
                opus_files.append(os.path.join(root, file))

    if not opus_files:
        print("No opus files found.")
        return

    print(f"Found {len(opus_files)} opus files. Checking integrity using 8 parallel processes...\n")

    # Prepare files with indices for parallel processing
    files_with_indices = list(enumerate(opus_files, 1))

    # Create partial function with expected_durations
    check_func = partial(check_file_worker, expected_durations=expected_durations)

    # Process files in parallel
    corrupted_files = []
    suspicious_files = []
    truncated_files = []
    valid_files = 0
    total_duration = 0
    results = []

    print("Processing files...")

    with Pool(processes=8) as pool:
        if TQDM_AVAILABLE:
            # Use tqdm progress bar
            with tqdm(total=len(files_with_indices), desc="Checking files", unit="file",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for result in pool.imap_unordered(check_func, files_with_indices):
                    results.append(result)
                    # Update progress bar with current file info
                    _, filepath, status, _, _, _ = result
                    video_id = os.path.splitext(os.path.basename(filepath))[0]
                    status_emoji = {"valid": "✓", "corrupted": "❌", "truncated": "⚠️", "suspicious": "⚠️"}.get(status, "?")
                    pbar.set_postfix_str(f"{status_emoji} {video_id[:30]}", refresh=False)
                    pbar.update(1)
        else:
            # Fallback without progress bar
            print("(Install 'tqdm' for progress bar: pip install tqdm)")
            for i, result in enumerate(pool.imap_unordered(check_func, files_with_indices), 1):
                results.append(result)
                if i % 100 == 0:  # Print progress every 100 files
                    print(f"  Processed {i}/{len(files_with_indices)} files...")

    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])

    # Quick summary after progress bar
    problem_count = sum(1 for r in results if r[2] != 'valid')
    if TQDM_AVAILABLE:
        print(f"\n✓ Checked {len(results)} files. Found {problem_count} problematic files.")

    # Process results
    if not summary_only:
        print("\nDetailed Results:")
        print("-" * 80)

    for i, filepath, status, actual_duration, expected_duration, extra_info in results:
        video_id = os.path.splitext(os.path.basename(filepath))[0]

        if status == 'corrupted':
            corrupted_files.append((filepath, extra_info))
            if not summary_only:
                print(f"[{i:4d}/{len(opus_files)}] {video_id:.<50} ❌ CORRUPTED - {extra_info}")
        elif status == 'truncated':
            truncated_files.append((filepath, actual_duration, expected_duration, extra_info))
            if not summary_only:
                print(f"[{i:4d}/{len(opus_files)}] {video_id:.<50} ⚠️  TRUNCATED - Expected {format_duration(expected_duration)}, got {format_duration(actual_duration)} ({extra_info:.1f}% off)")
        elif status == 'suspicious':
            suspicious_files.append((filepath, actual_duration))
            if not summary_only:
                print(f"[{i:4d}/{len(opus_files)}] {video_id:.<50} ⚠️  SUSPICIOUS - Very short ({format_duration(actual_duration)})")
        else:  # valid
            valid_files += 1
            total_duration += actual_duration
            if not summary_only:
                if expected_duration:
                    print(f"[{i:4d}/{len(opus_files)}] {video_id:.<50} ✓ OK ({format_duration(actual_duration)})")
                else:
                    print(f"[{i:4d}/{len(opus_files)}] {video_id:.<50} ✓ OK ({format_duration(actual_duration)}) [No expected duration]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total files checked: {len(opus_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Truncated files (duration mismatch): {len(truncated_files)}")
    print(f"Suspicious files (very short): {len(suspicious_files)}")

    if valid_files > 0:
        print(f"Total duration of valid files: {format_duration(total_duration)}")
        print(f"Average duration: {format_duration(total_duration / valid_files)}")

    if corrupted_files:
        print("\n❌ CORRUPTED FILES:")
        for filepath, error in corrupted_files:
            print(f"  - {filepath}")
            print(f"    Error: {error}")

    if truncated_files:
        print("\n⚠️  TRUNCATED FILES (significant duration mismatch - likely interrupted):")
        for filepath, actual, expected, diff_percent in truncated_files:
            print(f"  - {filepath}")
            print(f"    Expected: {format_duration(expected)}, Got: {format_duration(actual)} ({diff_percent:.1f}% off)")

    if suspicious_files:
        print("\n⚠️  SUSPICIOUS FILES (very short, no expected duration):")
        for filepath, duration in suspicious_files:
            print(f"  - {filepath} (duration: {format_duration(duration)})")

    # Check for orphaned m4a files (might indicate failed conversions)
    print("\n" + "=" * 60)
    print("Checking for unconverted m4a files...")
    m4a_files = []
    for root, _, files in os.walk(download_dir):
        for file in files:
            if file.endswith('.m4a'):
                m4a_files.append(os.path.join(root, file))

    if m4a_files:
        print(f"\n⚠️  Found {len(m4a_files)} unconverted m4a files:")
        for filepath in m4a_files:
            print(f"  - {filepath}")
        print("These files may indicate interrupted conversions.")

    # Print all problematic file paths for easy processing
    all_problematic = []

    # Collect all problematic opus files
    for filepath, _ in corrupted_files:
        all_problematic.append(filepath)

    for filepath, _, _, _ in truncated_files:
        all_problematic.append(filepath)

    for filepath, _ in suspicious_files:
        all_problematic.append(filepath)

    if all_problematic:
        if not summary_only:
            print("\n" + "=" * 60)
            print("ALL PROBLEMATIC OPUS FILES (for easy copying/processing):")
            print("-" * 60)
            for filepath in sorted(all_problematic):
                print(filepath)
            print("-" * 60)
            print(f"Total problematic files: {len(all_problematic)}")

        # Always print video IDs for easy re-downloading
        print("\n" + "=" * 60)
        print("VIDEO IDs TO RE-DOWNLOAD:")
        print("-" * 60)
        video_ids = set()
        for filepath in all_problematic:
            video_id = os.path.splitext(os.path.basename(filepath))[0]
            video_ids.add(video_id)
        for video_id in sorted(video_ids):
            print(video_id)
        print("-" * 60)
        print(f"Total videos to re-download: {len(video_ids)}")

        return all_problematic, video_ids

    return [], set()


def delete_problematic_files_and_update_csv(problematic_files, video_ids, csv_file):
    """
    Delete problematic files and update the CSV to mark them as not downloaded.
    """
    print("\n" + "=" * 60)
    print("CLEANUP OPERATION")
    print("=" * 60)

    # Confirm with user
    print("\nThis will:")
    print(f"1. Delete {len(problematic_files)} problematic opus files")
    print(f"2. Mark {len(video_ids)} videos as not downloaded in {csv_file}")

    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return

    # Delete files
    deleted_count = 0
    print("\nDeleting files...")
    for filepath in problematic_files:
        try:
            os.remove(filepath)
            print(f"✓ Deleted: {filepath}")
            deleted_count += 1
        except (OSError, IOError) as e:
            print(f"✗ Failed to delete {filepath}: {e}")

    print(f"\nDeleted {deleted_count}/{len(problematic_files)} files.")

    # Update CSV
    print(f"\nUpdating {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        original_count = df['downloaded'].sum()

        # Mark videos as not downloaded
        updated_count = 0
        for _, row in df.iterrows():
            video_id = get_video_id(row['url'])
            if video_id in video_ids:
                df.loc[df['url'] == row['url'], 'downloaded'] = False
                updated_count += 1

        # Save the updated CSV
        df.to_csv(csv_file, index=False)
        new_count = df['downloaded'].sum()

        print("✓ Updated CSV successfully")
        print(f"  - Marked {updated_count} videos as not downloaded")
        print(f"  - Downloaded count: {original_count} → {new_count}")

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IOError) as e:
        print(f"✗ Failed to update CSV: {e}")


def main():
    # Check if ffprobe is available
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffprobe not found. Please install ffmpeg.")
        print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Check integrity of downloaded audio files and optionally clean up problematic files.'
    )
    parser.add_argument(
        'download_dir',
        nargs='?',
        default=DOWNLOAD_DIR,
        help='Directory containing downloaded files (default: {DOWNLOAD_DIR})'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=CSV_FILE,
        help='CSV file with video metadata (default: {CSV_FILE})'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete problematic files and update CSV'
    )
    parser.add_argument(
        '--auto-yes',
        action='store_true',
        help='Automatically answer yes to cleanup confirmation'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary without detailed file-by-file results'
    )

    args = parser.parse_args()

    print(f"Using download directory: {args.download_dir}")
    print(f"Using CSV file: {args.csv_file}\n")

    # Run the integrity check
    problematic_files, video_ids = check_download_directory(args.download_dir, args.csv_file, args.summary_only)

    # If cleanup is requested and there are problematic files
    if args.cleanup and problematic_files:
        if args.auto_yes:
            # Skip confirmation prompt
            print("\n" + "=" * 60)
            print("CLEANUP OPERATION (auto-yes mode)")
            print("=" * 60)

            # Delete files
            deleted_count = 0
            print("\nDeleting files...")
            for filepath in problematic_files:
                try:
                    os.remove(filepath)
                    print(f"✓ Deleted: {filepath}")
                    deleted_count += 1
                except (OSError, IOError) as e:
                    print(f"✗ Failed to delete {filepath}: {e}")

            print(f"\nDeleted {deleted_count}/{len(problematic_files)} files.")

            # Update CSV
            print(f"\nUpdating {args.csv_file}...")
            try:
                df = pd.read_csv(args.csv_file)
                original_count = df['downloaded'].sum()

                # Mark videos as not downloaded
                updated_count = 0
                for _, row in df.iterrows():
                    video_id = get_video_id(row['url'])
                    if video_id in video_ids:
                        df.loc[df['url'] == row['url'], 'downloaded'] = False
                        updated_count += 1

                # Save the updated CSV
                df.to_csv(args.csv_file, index=False)
                new_count = df['downloaded'].sum()

                print("✓ Updated CSV successfully")
                print(f"  - Marked {updated_count} videos as not downloaded")
                print(f"  - Downloaded count: {original_count} → {new_count}")

            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IOError) as e:
                print(f"✗ Failed to update CSV: {e}")
        else:
            delete_problematic_files_and_update_csv(problematic_files, video_ids, args.csv_file)
    elif args.cleanup and not problematic_files:
        print("\n✓ No problematic files found. Nothing to clean up!")


if __name__ == "__main__":
    main()
