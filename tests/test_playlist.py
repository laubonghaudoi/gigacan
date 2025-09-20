from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gigacan.playlist import (
    PlaylistItem,
    export_playlist_metadata_csv,
    fetch_playlist_metadata,
    write_playlist_metadata_csv,
)


class DummyYoutubeDL:
    def __init__(self, options: dict[str, Any]):
        self.options = options
        self._enter_count = 0

    def __enter__(self) -> "DummyYoutubeDL":
        self._enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context cleanup is trivial
        return None

    def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
        assert download is False
        if url == "https://playlist":
            return {
                "_type": "playlist",
                "entries": [
                    {
                        "webpage_url": "https://example.com/1",
                        "title": "First",
                        "description": "Desc 1",
                        "duration": 61,
                    },
                    {
                        "_type": "url",
                        "url": "https://example.com/2",
                    },
                    None,
                ],
            }
        if url == "https://channel":
            return {
                "_type": "channel",
                "channel_id": "UC123",
                "entries": [],
            }
        if url == "https://www.youtube.com/playlist?list=UU123":
            return {
                "_type": "playlist",
                "entries": [
                    {
                        "webpage_url": "https://example.com/c1",
                        "title": "Channel Video",
                        "description": "Channel description",
                        "duration": 123,
                    }
                ],
            }
        if url == "https://example.com/2":
            return {
                "webpage_url": "https://example.com/2",
                "title": "Second",
                "description": "Desc 2",
                "duration": 42,
            }
        raise AssertionError(f"Unexpected URL requested: {url}")


class EmptyPlaylistDL(DummyYoutubeDL):
    def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
        assert download is False
        return {"_type": "playlist", "entries": []}


@pytest.fixture(autouse=True)
def _restore_dl(monkeypatch):
    monkeypatch.setattr("gigacan.playlist.YoutubeDL", DummyYoutubeDL)


def test_fetch_playlist_metadata_returns_entries():
    items = fetch_playlist_metadata("https://playlist")

    assert [item.url for item in items] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert items[0].title == "First"
    assert items[1].description == "Desc 2"
    assert items[1].duration_seconds == 42


def test_fetch_playlist_metadata_raises_when_no_entries(monkeypatch):
    monkeypatch.setattr("gigacan.playlist.YoutubeDL", EmptyPlaylistDL)

    with pytest.raises(ValueError):
        fetch_playlist_metadata("https://playlist")


def test_fetch_playlist_metadata_channel_fallback():
    progress_calls: list[tuple[int, int]] = []
    status_messages: list[str] = []

    def progress(done: int, total: int) -> None:
        progress_calls.append((done, total))

    def status(message: str) -> None:
        status_messages.append(message)

    items = fetch_playlist_metadata(
        "https://channel",
        progress_callback=progress,
        status_callback=status,
    )

    assert [item.url for item in items] == ["https://example.com/c1"]
    assert items[0].description == "Channel description"
    assert items[0].duration_seconds == 123
    assert status_messages == [
        "Falling back to channel uploads playlist for metadata collection."
    ]
    assert progress_calls[0] == (0, 0)
    assert progress_calls[-1] == (1, 1)


def test_write_playlist_metadata_csv(tmp_path: Path):
    destination = tmp_path / "metadata.csv"
    entries = [
        PlaylistItem(
            url="https://example.com/1",
            title="Title 1",
            description="Desc",
            duration_seconds=10,
        ),
        PlaylistItem(
            url="https://example.com/2",
            title="Title 2",
            description="Desc 2",
            duration_seconds=None,
        ),
    ]

    write_playlist_metadata_csv(entries, destination)

    content = destination.read_text(encoding="utf-8")
    assert content.splitlines() == [
        "url,title,description,duration_seconds",
        "https://example.com/1,Title 1,Desc,10",
        "https://example.com/2,Title 2,Desc 2,",
    ]


def test_export_playlist_metadata_csv(tmp_path: Path, monkeypatch):
    entries = [
        PlaylistItem(
            url="https://example.com/1",
            title="First",
            description="First desc",
            duration_seconds=1,
        ),
        PlaylistItem(
            url="https://example.com/2",
            title="Second",
            description="Second desc",
            duration_seconds=2,
        ),
    ]

    monkeypatch.setattr(
        "gigacan.playlist.fetch_playlist_metadata", lambda *a, **k: entries
    )

    destination = tmp_path / "out.csv"
    count = export_playlist_metadata_csv("https://playlist", destination)

    assert count == len(entries)
    assert (
        destination.read_text(encoding="utf-8").splitlines()[0]
        == "url,title,description,duration_seconds"
    )
