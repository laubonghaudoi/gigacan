import csv
import importlib
import os
from typing import Callable, Protocol, TypeGuard, TypedDict, cast

# --- CONFIGURATION ---
# 1. Make sure to set the YOUTUBE_API_KEY environment variable.
API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# 2. The username of the channel
CHANNEL_USERNAME = "@legcogovhk"

# 3. The name of the output CSV file
CSV_FILENAME = "legco.csv"
# --- END CONFIGURATION ---


class VideoMetadata(TypedDict):
    url: str
    title: str
    description: str
    publish_date: str
    duration: str
    downloaded: str


class _ListRequest(Protocol):
    def execute(self) -> dict[str, object]: ...


class _ChannelsResource(Protocol):
    def list(self, *, part: str, forHandle: str) -> _ListRequest: ...


class _PlaylistItemsResource(Protocol):
    def list(
        self,
        *,
        part: str,
        playlistId: str,
        maxResults: int,
        pageToken: str | None,
    ) -> _ListRequest: ...


class _VideosResource(Protocol):
    def list(self, *, part: str, id: str) -> _ListRequest: ...


class YouTubeClient(Protocol):
    def channels(self) -> _ChannelsResource: ...

    def playlistItems(self) -> _PlaylistItemsResource: ...

    def videos(self) -> _VideosResource: ...


def parse_duration(duration_str: str) -> str:
    """
    Parse an ISO 8601 duration fragment (e.g., "PT1H2M3S") into HH:MM:SS.
    """

    if not duration_str.startswith("PT"):
        return "00:00:00"

    duration_str = duration_str[2:]
    hours, minutes, seconds = 0, 0, 0

    if "H" in duration_str:
        parts = duration_str.split("H")
        hours = int(parts[0])
        duration_str = parts[1]
    if "M" in duration_str:
        parts = duration_str.split("M")
        minutes = int(parts[0])
        duration_str = parts[1]
    if "S" in duration_str:
        seconds = int(duration_str.replace("S", ""))

    return f"{hours:02}:{minutes:02}:{seconds:02}"


BuildFunction = Callable[..., object]


def get_youtube_client(api_key: str) -> YouTubeClient:
    discovery_module = importlib.import_module("googleapiclient.discovery")
    build_func = cast(BuildFunction, getattr(discovery_module, "build"))
    return cast(YouTubeClient, build_func("youtube", "v3", developerKey=api_key))


def _is_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_dict_list(value: object) -> TypeGuard[list[dict[str, object]]]:
    if not isinstance(value, list):
        return False
    for element in cast(list[object], value):
        if not isinstance(element, dict):
            return False
    return True


def extract_uploads_playlist_id(
    channel_username: str, youtube: YouTubeClient
) -> str | None:
    channel_response = (
        youtube.channels()
        .list(part="contentDetails", forHandle=channel_username)
        .execute()
    )

    channel_items = channel_response.get("items", [])
    if not _is_dict_list(channel_items) or not channel_items:
        return None

    content_details = channel_items[0].get("contentDetails")
    if not _is_dict(content_details):
        return None

    related_playlists = content_details.get("relatedPlaylists")
    if not _is_dict(related_playlists):
        return None

    uploads_playlist_id = related_playlists.get("uploads")
    return uploads_playlist_id if isinstance(uploads_playlist_id, str) else None


def fetch_all_video_ids(playlist_id: str, youtube: YouTubeClient) -> list[str]:
    video_ids: list[str] = []
    next_page_token: str | None = None

    while True:
        playlist_response = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            .execute()
        )

        playlist_items = playlist_response.get("items", [])
        if _is_dict_list(playlist_items):
            for item in playlist_items:
                content_details = item.get("contentDetails")
                if not _is_dict(content_details):
                    continue
                video_id = content_details.get("videoId")
                if isinstance(video_id, str):
                    video_ids.append(video_id)

        token = playlist_response.get("nextPageToken")
        next_page_token = token if isinstance(token, str) else None
        if not next_page_token:
            break

    return video_ids


def fetch_video_details(
    video_ids: list[str], youtube: YouTubeClient
) -> list[VideoMetadata]:
    details: list[VideoMetadata] = []

    for index in range(0, len(video_ids), 50):
        video_batch_ids = video_ids[index : index + 50]
        video_response = (
            youtube.videos()
            .list(part="snippet,contentDetails", id=",".join(video_batch_ids))
            .execute()
        )

        response_items = video_response.get("items", [])
        if not _is_dict_list(response_items):
            continue

        for item in response_items:
            snippet_value = item.get("snippet")
            snippet = snippet_value if _is_dict(snippet_value) else {}

            published_at = snippet.get("publishedAt")
            publish_date = (
                published_at.split("T")[0]
                if isinstance(published_at, str) and "T" in published_at
                else ""
            )

            description = snippet.get("description")
            description_text = description if isinstance(description, str) else ""
            description_safe = (
                description_text.replace("\r\n", "\\n")
                .replace("\n", "\\n")
                .replace("\r", "\\n")
            )

            content_details = item.get("contentDetails")
            duration_str = ""
            if _is_dict(content_details):
                duration_value = content_details.get("duration")
                if isinstance(duration_value, str):
                    duration_str = duration_value

            video_id = item.get("id")
            if not isinstance(video_id, str):
                continue

            title_value = snippet.get("title")
            title = title_value if isinstance(title_value, str) else ""

            details.append(
                VideoMetadata(
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    title=title,
                    description=description_safe,
                    publish_date=publish_date,
                    duration=parse_duration(duration_str),
                    downloaded="false",
                )
            )

    return details


def write_metadata_to_csv(metadata: list[VideoMetadata], filename: str) -> None:
    fieldnames = [
        "url",
        "title",
        "description",
        "publish_date",
        "duration",
        "downloaded",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def main() -> None:
    if not API_KEY:
        print("ERROR: Please set the YOUTUBE_API_KEY environment variable.")
        return

    # Initialize the YouTube API service
    youtube = get_youtube_client(API_KEY)

    print(f"Fetching channel details for {CHANNEL_USERNAME}...")

    uploads_playlist_id = extract_uploads_playlist_id(CHANNEL_USERNAME, youtube)
    if uploads_playlist_id is None:
        print(f"Error: Could not find a channel with the handle {CHANNEL_USERNAME}")
        return

    print(f"Found uploads playlist ID: {uploads_playlist_id}")

    # 2. Get all video IDs from the uploads playlist
    print("Fetching all video IDs from the playlist (this may take a moment)...")
    all_video_ids = fetch_all_video_ids(uploads_playlist_id, youtube)

    print(f"Total videos found: {len(all_video_ids)}")
    print("Fetching detailed metadata for all videos (in batches of 50)...")

    all_video_details = fetch_video_details(all_video_ids, youtube)

    print(f"Writing data to {CSV_FILENAME}...")
    write_metadata_to_csv(all_video_details, CSV_FILENAME)

    print("Done! Your metadata has been saved.")


if __name__ == "__main__":
    main()
