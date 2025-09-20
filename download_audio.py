import subprocess
import pandas as pd
from urllib.parse import urlparse, parse_qs
import os

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
    output_filename = os.path.join('download', f'{video_id}.opus')
    
    command = [
        'yt-dlp',
        '-o', output_filename,
        '-x',
        '--audio-format', 'opus',
        '--postprocessor-args', '-ar 16000',
        '--cookies', 'cookies.txt',
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

def main():
    csv_file = 'legco.csv'
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    for index, row in df.iterrows():
        if not row['downloaded']:
            url = row['url']
            video_id = get_video_id(url)
            
            if video_id:
                success = download_audio(video_id, url)
                if success:
                    df.loc[index, 'downloaded'] = True
                    # Save after each successful download to prevent data loss
                    df.to_csv(csv_file, index=False)
            else:
                print(f"Could not extract video ID from URL: {url}")

    print("All videos have been processed.")

if __name__ == "__main__":
    main()
