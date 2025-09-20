"""Utilities for retrieving metadata about YouTube playlists."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    from yt_dlp import YoutubeDL  # type: ignore
except ImportError as exc:  # pragma: no cover - import guarded for runtime
    raise RuntimeError("yt-dlp must be installed to fetch playlist metadata") from exc


@dataclass(slots=True)
class PlaylistItem:
    """Container for the metadata we export for each playlist entry."""

    url: str
    title: str
    description: str
    duration_seconds: int | None


ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


def fetch_playlist_metadata(
    playlist_url: str,
    *,
    ydl_options: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> list[PlaylistItem]:
    """Return metadata for every accessible entry in a YouTube playlist.

    Args:
        playlist_url: Public, unlisted, or channel YouTube URL.
        ydl_options: Extra options forwarded to :class:`yt_dlp.YoutubeDL`.
        progress_callback: Optional callable receiving ``(completed, total)`` counts
            while entries are processed. ``total`` may be ``0`` while the playlist is
            being resolved.
        status_callback: Optional callable accepting human-readable status updates
            (e.g. when falling back to a channel uploads playlist).

    Returns:
        A list of :class:`PlaylistItem` objects in playlist order.

    Raises:
        ValueError: If no playable entries are found.
        RuntimeError: If ``yt-dlp`` fails to extract the playlist.
    """

    options: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": True,
        "noplaylist": False,
    }
    if ydl_options:
        options.update(ydl_options)

    with YoutubeDL(options) as ydl:
        info_dict = _extract_with_error(
            ydl,
            playlist_url,
            status_callback=status_callback,
        )
        filtered_items = _collect_items(
            info_dict,
            ydl,
            progress_callback,
            status_callback,
        )

        if filtered_items:
            return filtered_items

        fallback_url = _uploads_playlist_from_info(info_dict)
        if fallback_url and fallback_url != playlist_url:
            if status_callback:
                status_callback(
                    "Falling back to channel uploads playlist for metadata collection."
                )
            info_dict = _extract_with_error(
                ydl,
                fallback_url,
                status_callback=status_callback,
            )
            filtered_items = _collect_items(
                info_dict,
                ydl,
                progress_callback,
                status_callback,
            )
            if filtered_items:
                return filtered_items

    raise ValueError(f"No playable videos found for playlist: {playlist_url}")


def write_playlist_metadata_csv(
    entries: Sequence[PlaylistItem],
    destination: Path | str,
) -> None:
    """Write playlist metadata to a CSV file with url/title/description/duration."""

    if not entries:
        return

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["url", "title", "description", "duration_seconds"])
        for item in entries:
            duration = "" if item.duration_seconds is None else str(item.duration_seconds)
            writer.writerow([item.url, item.title, item.description, duration])


def export_playlist_metadata_csv(
    playlist_url: str,
    destination: Path | str,
    *,
    ydl_options: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> int:
    """Fetch metadata for ``playlist_url`` and write it to ``destination``.

    Returns the number of entries written to the CSV file.

    ``progress_callback`` and ``status_callback`` mirror the parameters documented in
    :func:`fetch_playlist_metadata`.
    """

    entries = fetch_playlist_metadata(
        playlist_url,
        ydl_options=ydl_options,
        progress_callback=progress_callback,
        status_callback=status_callback,
    )
    write_playlist_metadata_csv(entries, destination)
    return len(entries)


def _iter_entries(info_dict: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    if not info_dict:
        return

    entries = info_dict.get("entries")
    if entries is None:
        yield info_dict
        return

    for entry in entries or []:
        if entry:
            yield entry


def _ensure_full_entry(
    raw_entry: dict[str, Any],
    ydl: YoutubeDL,
    *,
    status_callback: StatusCallback | None = None,
) -> dict[str, Any]:
    if raw_entry.get("_type") == "url" and raw_entry.get("url"):
        extracted = _retry_extract(
            ydl,
            raw_entry["url"],
            allow_failure=True,
            status_callback=status_callback,
        )
        if extracted is not None:
            return extracted
        if status_callback:
            status_callback(
                "Failed to fully resolve a playlist entry after multiple retries; "
                "using partial metadata instead."
            )
    return raw_entry


def _coerce_playlist_item(entry: dict[str, Any] | None) -> PlaylistItem | None:
    if not entry:
        return None

    url = _coerce_text(
        entry.get("webpage_url")
        or entry.get("original_url")
        or entry.get("url")
    )

    if not url:
        video_id = _coerce_text(entry.get("id"))
        if not video_id:
            return None
        url = f"https://www.youtube.com/watch?v={video_id}"

    title = _coerce_text(entry.get("title"))
    description = _coerce_text(entry.get("description"))
    duration = _extract_duration_seconds(entry)

    return PlaylistItem(
        url=url,
        title=title,
        description=description,
        duration_seconds=duration,
    )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _coerce_duration(value: Any) -> int | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number < 0 or not number == number:  # NaN check
        return None

    return int(number)


def _extract_duration_seconds(entry: dict[str, Any]) -> int | None:
    for key in ("duration", "duration_seconds"):
        duration = _coerce_duration(entry.get(key))
        if duration is not None:
            return duration

    duration_ms = entry.get("duration_ms")
    if duration_ms is not None:
        try:
            millis = float(duration_ms)
        except (TypeError, ValueError):
            millis = None
        if millis is not None and millis >= 0:
            return int(millis / 1000)

    duration_str = _coerce_text(entry.get("duration_string"))
    if duration_str:
        parts = duration_str.split(":")
        try:
            seconds = 0
            for part in parts:
                seconds = seconds * 60 + int(part)
        except ValueError:
            return None
        return seconds

    return None


def _retry_extract(
    ydl: YoutubeDL,
    url: str,
    *,
    allow_failure: bool = False,
    status_callback: StatusCallback | None = None,
    max_attempts: int = 5,
    initial_delay: float = 5.0,
    max_delay: float = 60.0,
) -> dict[str, Any] | None:
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return ydl.extract_info(url, download=False)
        except Exception as exc:  # pragma: no cover - depends on network state
            last_exc = exc
            if attempt == max_attempts:
                break
            if status_callback:
                status_callback(
                    f"Attempt {attempt}/{max_attempts} failed; retrying in {int(delay)}s."
                )
            time.sleep(delay)
            delay = min(max_delay, delay * 2)

    if allow_failure:
        return None

    if status_callback:
        status_callback(
            "Failed to fetch metadata after multiple retries; aborting extraction."
        )

    if last_exc is not None:
        raise RuntimeError(f"Failed to fetch metadata for: {url}") from last_exc
    raise RuntimeError(f"Failed to fetch metadata for: {url}")


def _extract_with_error(
    ydl: YoutubeDL,
    url: str,
    *,
    status_callback: StatusCallback | None = None,
) -> dict[str, Any]:
    result = _retry_extract(
        ydl,
        url,
        status_callback=status_callback,
    )
    if result is None:
        raise RuntimeError(f"Failed to fetch metadata for playlist: {url}")
    return result


def _collect_items(
    info_dict: dict[str, Any],
    ydl: YoutubeDL,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> list[PlaylistItem]:
    entries = list(_iter_entries(info_dict))
    total = len(entries)

    if progress_callback:
        progress_callback(0, total)

    collected: list[PlaylistItem] = []
    for index, raw_entry in enumerate(entries, start=1):
        item = _coerce_playlist_item(
            _ensure_full_entry(
                raw_entry,
                ydl,
                status_callback=status_callback,
            )
        )
        if item is not None:
            collected.append(item)
        if progress_callback:
            progress_callback(index, total)

    return collected


def _uploads_playlist_from_info(info_dict: dict[str, Any]) -> str | None:
    playlist_id = _coerce_text(info_dict.get("playlist_id"))
    if playlist_id.startswith("UU"):
        return f"https://www.youtube.com/playlist?list={playlist_id}"

    channel_id = _coerce_text(info_dict.get("channel_id"))
    if channel_id.startswith("UC"):
        uploads_id = "UU" + channel_id[2:]
        return f"https://www.youtube.com/playlist?list={uploads_id}"

    return None


__all__ = [
    "PlaylistItem",
    "fetch_playlist_metadata",
    "write_playlist_metadata_csv",
    "export_playlist_metadata_csv",
]
