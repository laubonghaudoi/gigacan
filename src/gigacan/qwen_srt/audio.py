from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


TARGET_SAMPLE_RATE = 16000


def load_audio_mono_16k(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono 16kHz float32 samples."""
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
            sample_rate = TARGET_SAMPLE_RATE

        audio = waveform.squeeze(0).to(dtype=torch.float32).contiguous().cpu().numpy()
        return audio, sample_rate
    except Exception:
        # TorchCodec may be unavailable in some environments; decode once via ffmpeg.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        pcm16 = np.frombuffer(proc.stdout, dtype=np.int16)
        audio = (pcm16.astype(np.float32) / 32768.0).copy()
        return audio, TARGET_SAMPLE_RATE


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
