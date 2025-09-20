import os
import pandas as pd
import shutil
from urllib.parse import urlparse, parse_qs

# --- CONFIGURATION ---
CSV_FILE = 'legco_20250920.csv'
DOWNLOAD_DIR = 'download/'
# --- END CONFIGURATION ---

def get_video_id(url):
    """
    Extracts the YouTube video ID from a URL.
    Handles standard youtube.com and youtu.be links.
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
    except Exception as e:
        print(f"Could not parse URL '{url}': {e}")
    return None

def organize_downloads_by_year():
    """
    Organizes downloaded audio files into subdirectories based on their
    upload year, sourced from the CSV file.
    """
    print(f"Reading video metadata from '{CSV_FILE}'...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 1. Create a mapping from video_id to its upload year
    video_id_to_year = {}
    for _, row in df.iterrows():
        video_id = get_video_id(row['url'])
        if video_id and pd.notna(row['publish_date']):
            # Extract year from 'YYYY-MM-DD' format
            year = str(row['publish_date']).split('-')[0]
            video_id_to_year[video_id] = year
    
    print(f"Created a map for {len(video_id_to_year)} videos.")

    # 2. Get list of .opus files in the download directory
    if not os.path.isdir(DOWNLOAD_DIR):
        print(f"Error: Download directory '{DOWNLOAD_DIR}' not found.")
        return
        
    try:
        files_to_organize = [
            f for f in os.listdir(DOWNLOAD_DIR)
            if f.endswith('.opus') and os.path.isfile(os.path.join(DOWNLOAD_DIR, f))
        ]
        if not files_to_organize:
            print("No .opus files found in the download directory to organize.")
            return
    except Exception as e:
        print(f"Error listing files in '{DOWNLOAD_DIR}': {e}")
        return

    # 3. Move files to their respective year folders
    moved_count = 0
    unmapped_count = 0
    print(f"Starting to organize {len(files_to_organize)} files...")

    for filename in files_to_organize:
        video_id = os.path.splitext(filename)[0]
        year = video_id_to_year.get(video_id)

        if year:
            source_path = os.path.join(DOWNLOAD_DIR, filename)
            year_dir = os.path.join(DOWNLOAD_DIR, year)
            destination_path = os.path.join(year_dir, filename)

            # Create the year-specific subdirectory if it doesn't exist
            os.makedirs(year_dir, exist_ok=True)

            try:
                shutil.move(source_path, destination_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving '{filename}': {e}")
        else:
            print(f"Warning: No upload year found for video ID '{video_id}'. Skipping.")
            unmapped_count += 1
    
    print("\n--- Organization Complete ---")
    print(f"Successfully moved {moved_count} files.")
    if unmapped_count > 0:
        print(f"{unmapped_count} files were skipped as they could not be mapped to a year.")

if __name__ == "__main__":
    organize_downloads_by_year()
