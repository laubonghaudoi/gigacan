import os
import pandas as pd
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

def scan_and_update_progress():
    """
    Scans the download directory and updates the CSV file to reflect
    the actual download progress.
    """
    print(f"Scanning for downloaded files in '{DOWNLOAD_DIR}'...")

    if not os.path.isdir(DOWNLOAD_DIR):
        print(f"Error: Download directory '{DOWNLOAD_DIR}' not found.")
        return

    # 1. Get a set of video IDs from the filenames in the download directory
    try:
        downloaded_files = os.listdir(DOWNLOAD_DIR)
        downloaded_ids = {os.path.splitext(f)[0] for f in downloaded_files if f.endswith('.opus')}
        print(f"Found {len(downloaded_ids)} downloaded audio files.")
    except Exception as e:
        print(f"Error reading directory '{DOWNLOAD_DIR}': {e}")
        return

    # 2. Read the CSV file
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file '{CSV_FILE}': {e}")
        return

    # 3. Iterate and update the 'downloaded' status
    updates_made = 0
    for index, row in df.iterrows():
        # Only check rows that are currently marked as not downloaded
        if row['downloaded'] is False:
            video_id = get_video_id(row['url'])
            if video_id and video_id in downloaded_ids:
                df.loc[index, 'downloaded'] = True
                updates_made += 1

    # 4. Save the updated DataFrame back to the CSV
    if updates_made > 0:
        print(f"Updating {updates_made} rows in the CSV to 'downloaded: True'.")
        try:
            df.to_csv(CSV_FILE, index=False)
            print(f"Successfully updated '{CSV_FILE}'.")
        except Exception as e:
            print(f"Error saving updated CSV file: {e}")
    else:
        print("No updates needed. CSV file is already in sync with the download folder.")

if __name__ == "__main__":
    scan_and_update_progress()
