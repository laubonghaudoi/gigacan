from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(path: Path, entries: list[tuple[int, int, str]]) -> None:
    """Write subtitle entries to ``path`` in SRT format."""
    chunks = []
    for idx, (start_ms, end_ms, text) in enumerate(entries, start=1):
        chunks.append(
            f"{idx}\n{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}\n{text}\n"
        )
    body = "\n".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(body)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def extract_segment_to_wav(
    audio_path: Path, start_ms: int, end_ms: int, out_wav_path: Path
) -> None:
    """Extract one timestamp slice as 16kHz mono PCM WAV via ffmpeg."""
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        raise ValueError(f"Invalid segment range: {start_ms} -> {end_ms}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav_path),
    ]
    subprocess.run(cmd, check=True)
