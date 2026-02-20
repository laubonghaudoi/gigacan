from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.vad_cache import load_vad_cache, save_vad_cache


def test_vad_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"dummy")
    segments = [(0, 1000), (1200, 3000)]

    save_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
        segments=segments,
    )

    loaded = load_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
    )

    assert loaded == segments


def test_vad_cache_invalidates_on_audio_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    audio = tmp_path / "b.opus"
    audio.write_bytes(b"dummy")
    save_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
        segments=[(0, 1000)],
    )

    audio.write_bytes(b"changed")

    loaded = load_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
    )

    assert loaded is None


def test_vad_cache_invalidates_on_parameters_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    audio = tmp_path / "c.opus"
    audio.write_bytes(b"dummy")
    save_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
        segments=[(0, 1000)],
    )

    loaded = load_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=500,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
    )

    assert loaded is None


def test_vad_cache_invalidates_on_end_silence_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    audio = tmp_path / "d.opus"
    audio.write_bytes(b"dummy")
    save_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=800,
        segments=[(0, 1000)],
    )

    loaded = load_vad_cache(
        cache_dir,
        audio,
        min_segment_ms=300,
        vad_max_segment_ms=15000,
        vad_max_end_silence_ms=600,
    )

    assert loaded is None
