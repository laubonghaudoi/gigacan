from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _cache_key(audio_path: Path) -> str:
    resolved = str(audio_path.resolve())
    return hashlib.sha1(resolved.encode("utf-8"), usedforsecurity=False).hexdigest()


def _cache_path(cache_dir: Path, audio_path: Path) -> Path:
    return cache_dir / f"{_cache_key(audio_path)}.json"


def load_vad_cache(
    cache_dir: Path,
    audio_path: Path,
    *,
    min_segment_ms: int,
    vad_max_segment_ms: int,
) -> list[tuple[int, int]] | None:
    path = _cache_path(cache_dir, audio_path)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        stat = audio_path.stat()
    except FileNotFoundError:
        return None

    if payload.get("size") != int(stat.st_size):
        return None
    if payload.get("mtime_ns") != int(stat.st_mtime_ns):
        return None
    if payload.get("min_segment_ms") != int(min_segment_ms):
        return None
    if payload.get("vad_max_segment_ms") != int(vad_max_segment_ms):
        return None

    raw = payload.get("segments", [])
    segments: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return None
        segments.append((int(item[0]), int(item[1])))
    return segments


def save_vad_cache(
    cache_dir: Path,
    audio_path: Path,
    *,
    min_segment_ms: int,
    vad_max_segment_ms: int,
    segments: list[tuple[int, int]],
) -> None:
    try:
        stat = audio_path.stat()
    except FileNotFoundError:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio": str(audio_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "min_segment_ms": int(min_segment_ms),
        "vad_max_segment_ms": int(vad_max_segment_ms),
        "segments": [[int(start), int(end)] for start, end in segments],
    }
    path = _cache_path(cache_dir, audio_path)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
