import subprocess
import pandas as pd
from urllib.parse import urlparse, parse_qs
import os
from multiprocessing import Pool

# --- CONFIGURATION ---
CSV_FILE = 'legco_20250920.csv'
DOWNLOAD_DIR = 'download/'
# --- END CONFIGURATION ---


def get_video_id(url):
    """
    Extracts the YouTube video ID from a URL.
    """
    if 'youtu.be' in url:
        return url.split('/')[-1]
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'www.youtube.com':
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
    return None


def download_audio(video_id, url):
    """
    Downloads audio from a YouTube URL and converts it to 16kHz opus format.
    The filename will be the video ID.
    """
    output_filename = os.path.join(DOWNLOAD_DIR, f'{video_id}.opus')

    command = [
        'yt-dlp',
        '-o', output_filename,
        '-x',
        '--audio-format', 'opus',
        '--postprocessor-args', '-ar 16000',
        '--cookies', './cookies.txt',
        url
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Audio downloaded and converted successfully: {output_filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading {url}: {e}")
        return False
    except FileNotFoundError:
        print("yt-dlp not found. Please ensure it is installed and in your PATH.")
        return False


def download_worker(task_data):
    """
    Worker function to download audio for a given URL and return the index for updating.
    """
    index, url = task_data
    video_id = get_video_id(url)
    if video_id:
        if download_audio(video_id, url):
            return index
    else:
        print(f"Could not extract video ID from URL: {url}")
    return None


def main():
    csv_file = CSV_FILE
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    # Sort the DataFrame by 'publish_date' in descending order
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df = df.sort_values(by='publish_date', ascending=False)

    tasks = []
    for index, row in df.iterrows():
        # Check if 'downloaded' is False or the column doesn't exist/is NaN
        if not row['downloaded']:
            tasks.append((index, row['url']))

    if not tasks:
        print("All videos have already been processed.")
        return

    print(f"Starting download of {len(tasks)} videos using 8 processes...")

    with Pool(processes=8) as pool:
        results = pool.map(download_worker, tasks)

    successful_downloads = 0
    for index in results:
        if index is not None:
            df.loc[index, 'downloaded'] = True
            successful_downloads += 1

    if successful_downloads > 0:
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with {successful_downloads} new downloads.")

    print("All processing is complete.")


if __name__ == "__main__":
    main()
