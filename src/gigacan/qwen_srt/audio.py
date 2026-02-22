from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16000
FFMPEG_CMD_BASE = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
]

VAD_CHUNK_SECONDS = 600
VAD_CHUNK_SAMPLES = VAD_CHUNK_SECONDS * TARGET_SAMPLE_RATE
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 4  # float32
MIN_VAD_SAMPLES = TARGET_SAMPLE_RATE // 2  # 0.5 s — minimum for FSMN feature extraction


def _ffmpeg_decode_cmd(audio_path: Path) -> list[str]:
    return [
        *FFMPEG_CMD_BASE,
        "-threads",
        "1",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]


def load_audio_mono_16k(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono 16kHz float32 samples.

    Reads ffmpeg output in chunks to avoid subprocess.run's internal
    buffering which doubles peak memory (subprocess buffer + numpy copy).
    """
    proc = subprocess.Popen(
        _ffmpeg_decode_cmd(audio_path),
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    chunks: list[np.ndarray] = []
    while True:
        raw = proc.stdout.read(VAD_CHUNK_BYTES)
        if not raw:
            break
        chunks.append(np.frombuffer(raw, dtype=np.float32).copy())
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio, TARGET_SAMPLE_RATE


def load_audio_and_vad_streaming(
    audio_path: Path,
    vad_model: Any,
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    """Decode audio and run VAD concurrently in a streaming fashion.

    A reader thread drains ffmpeg stdout into a queue while the caller
    thread runs VAD on each chunk. Total time ≈ max(decode, VAD).
    """
    from queue import Queue
    from threading import Thread

    proc = subprocess.Popen(
        _ffmpeg_decode_cmd(audio_path),
        stdout=subprocess.PIPE,
    )

    chunk_queue: Queue[np.ndarray | None] = Queue(maxsize=2)

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            while True:
                raw = proc.stdout.read(VAD_CHUNK_BYTES)
                if not raw:
                    break
                chunk_queue.put(np.frombuffer(raw, dtype=np.float32).copy())
        finally:
            chunk_queue.put(None)
            proc.wait()

    reader = Thread(target=_reader, daemon=True)
    reader.start()

    chunks: list[np.ndarray] = []
    raw_segments: list[tuple[int, int]] = []
    offset_ms = 0

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        chunks.append(chunk)

        if len(chunk) >= MIN_VAD_SAMPLES:
            vad_res = vad_model.generate(input=chunk)
            if vad_res and "value" in vad_res[0]:
                for start, end in vad_res[0]["value"]:
                    raw_segments.append(
                        (int(start) + offset_ms, int(end) + offset_ms)
                    )

        offset_ms += len(chunk) * 1000 // TARGET_SAMPLE_RATE

    reader.join()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return raw_segments, audio, TARGET_SAMPLE_RATE


def compute_vad_streaming(
    audio_path: Path,
    vad_model: Any,
) -> list[tuple[int, int]]:
    """Streaming decode+VAD that only returns segment boundaries.

    Unlike load_audio_and_vad_streaming, this does NOT accumulate the
    decoded audio in memory.  Peak RAM per call ≈ one 10-min chunk
    (~38 MB) regardless of file length.  Designed for multiprocessing
    pre-computation where the full waveform is not needed.
    """
    from queue import Queue
    from threading import Thread

    proc = subprocess.Popen(
        _ffmpeg_decode_cmd(audio_path),
        stdout=subprocess.PIPE,
    )

    chunk_queue: Queue[np.ndarray | None] = Queue(maxsize=2)

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            while True:
                raw = proc.stdout.read(VAD_CHUNK_BYTES)
                if not raw:
                    break
                chunk_queue.put(np.frombuffer(raw, dtype=np.float32).copy())
        finally:
            chunk_queue.put(None)
            proc.wait()

    reader = Thread(target=_reader, daemon=True)
    reader.start()

    raw_segments: list[tuple[int, int]] = []
    offset_ms = 0

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break

        if len(chunk) >= MIN_VAD_SAMPLES:
            vad_res = vad_model.generate(input=chunk)
            if vad_res and "value" in vad_res[0]:
                for start, end in vad_res[0]["value"]:
                    raw_segments.append(
                        (int(start) + offset_ms, int(end) + offset_ms)
                    )

        offset_ms += len(chunk) * 1000 // TARGET_SAMPLE_RATE
        del chunk

    reader.join()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    return raw_segments


def slice_audio_segment(
    audio: np.ndarray,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    """Slice a segment from preloaded audio samples."""
    if end_ms <= start_ms:
        raise ValueError(f"Invalid segment range: {start_ms} -> {end_ms}")

    start_idx = max(0, int(start_ms * sample_rate / 1000))
    end_idx = min(int(end_ms * sample_rate / 1000), audio.shape[0])
    if end_idx <= start_idx:
        raise ValueError(f"Invalid segment indices: {start_idx} -> {end_idx}")
    return np.asarray(audio[start_idx:end_idx], dtype=np.float32)
