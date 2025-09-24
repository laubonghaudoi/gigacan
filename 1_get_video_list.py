import os
import csv
from googleapiclient.discovery import build

# --- CONFIGURATION ---
# 1. Make sure to set the YOUTUBE_API_KEY environment variable.
API_KEY = ""

# 2. The username of the channel
CHANNEL_USERNAME = "@legcogovhk"

# 3. The name of the output CSV file
CSV_FILENAME = "legco.csv"
# --- END CONFIGURATION ---

# This function parses the strange duration format (e.g., "PT1H2M3S")


def parse_duration(duration_str):
    # This is a simple parser, you could make it more robust if needed
    if not duration_str.startswith('PT'):
        return "00:00:00"

    duration_str = duration_str[2:]
    hours, minutes, seconds = 0, 0, 0

    if 'H' in duration_str:
        parts = duration_str.split('H')
        hours = int(parts[0])
        duration_str = parts[1]
    if 'M' in duration_str:
        parts = duration_str.split('M')
        minutes = int(parts[0])
        duration_str = parts[1]
    if 'S' in duration_str:
        seconds = int(duration_str.replace('S', ''))

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main():
    if not API_KEY:
        print("ERROR: Please set the YOUTUBE_API_KEY environment variable.")
        return

    # Initialize the YouTube API service
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    print(f"Fetching channel details for {CHANNEL_USERNAME}...")

    # 1. Get the channel's "uploads" playlist ID
    channel_request = youtube.channels().list(
        part="contentDetails",
        forHandle=CHANNEL_USERNAME
    )
    channel_response = channel_request.execute()

    if not channel_response.get('items'):
        print(f"Error: Could not find a channel with the handle {CHANNEL_USERNAME}")
        return

    uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    print(f"Found uploads playlist ID: {uploads_playlist_id}")

    # 2. Get all video IDs from the uploads playlist
    all_video_ids = []
    next_page_token = None
    print("Fetching all video IDs from the playlist (this may take a moment)...")
    while True:
        playlist_request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,  # Max allowed per page
            pageToken=next_page_token
        )
        playlist_response = playlist_request.execute()

        all_video_ids.extend([item['contentDetails']['videoId'] for item in playlist_response['items']])

        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break
        print(f"  ...found {len(all_video_ids)} videos so far")

    print(f"Total videos found: {len(all_video_ids)}")
    print("Fetching detailed metadata for all videos (in batches of 50)...")

    # 3. Get video details (including duration) in batches of 50
    all_video_details = []
    for i in range(0, len(all_video_ids), 50):
        video_batch_ids = all_video_ids[i:i + 50]

        video_request = youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(video_batch_ids)
        )
        video_response = video_request.execute()

        for item in video_response['items']:
            snippet = item.get('snippet', {})
            published_at = snippet.get('publishedAt', '')
            description = snippet.get('description', '')
            # Preserve intentional newlines while encoding them safely for CSV consumers
            description_safe = (description
                                .replace('\r\n', '\\n')
                                .replace('\n', '\\n')
                                .replace('\r', '\\n'))

            all_video_details.append({
                'url': f"https://www.youtube.com/watch?v={item['id']}",
                'title': snippet.get('title', ''),
                'description': description_safe,
                'publish_date': published_at.split('T')[0] if published_at else '',
                'duration': parse_duration(item.get('contentDetails', {}).get('duration', '')),
                'downloaded': 'false'  # As requested
            })
        print(f"  ...processed {len(all_video_details)} of {len(all_video_ids)} videos")

    # 4. Write all data to the CSV file
    print(f"Writing data to {CSV_FILENAME}...")
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'description', 'publish_date', 'duration', 'downloaded']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(all_video_details)

    print("Done! Your metadata has been saved.")


if __name__ == "__main__":
    main()
