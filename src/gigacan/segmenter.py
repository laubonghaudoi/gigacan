import subprocess
from typing import List, Tuple


def check_ffmpeg() -> None:
    for cmd in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            raise RuntimeError(f"{cmd} is required but not found. Please install FFmpeg.") from e


def extract_mono_wav(input_path: str, wav_path: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(sr),
        "-f", "wav", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def try_silero_vad_segments(
    wav_path: str,
    max_seg_s: float,
    *,
    merge_gap_ms: int = 200,
    min_speech_ms: int = 200,
) -> List[Tuple[float, float]]:
    """
    Segment speech using Silero VAD, merging short pauses (<=0.2s),
    and split any resulting segment over max_seg_s.
    Returns list of (start_s, end_s). Empty list on failure (caller can fallback).
    """
    try:
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r".*torchaudio\._backend\.list_audio_backends.*",
            category=UserWarning,
        )
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

        model = load_silero_vad()
        wav = read_audio(wav_path)
        ts = get_speech_timestamps(
            wav,
            model,
            min_speech_duration_ms=int(min_speech_ms),
            # Use merge_gap_ms at VAD-level to minimize tiny splits
            min_silence_duration_ms=int(merge_gap_ms),
            speech_pad_ms=120,
            return_seconds=True,
        )

        segs_sec: List[Tuple[float, float]] = [
            (float(item["start"]), float(item["end"])) for item in ts
        ]
        segs_sec.sort(key=lambda x: x[0])

        # Merge adjacent segments with gaps <= merge_gap_ms
        merged_gap = max(0, int(merge_gap_ms)) / 1000.0
        merged: List[Tuple[float, float]] = []
        for seg in segs_sec:
            if not merged:
                merged.append(seg)
                continue
            prev_start, prev_end = merged[-1]
            cur_start, cur_end = seg
            if cur_start - prev_end <= merged_gap:
                merged[-1] = (prev_start, max(prev_end, cur_end))
            else:
                merged.append(seg)

        # Enforce max segment length only if positive; allow disabling via <= 0
        if max_seg_s is not None and max_seg_s > 0:
            bounded: List[Tuple[float, float]] = []
            for start_s, end_s in merged:
                cur = start_s
                while cur < end_s:
                    chunk_end = min(cur + max_seg_s, end_s)
                    bounded.append((cur, chunk_end))
                    cur = chunk_end
            return bounded
        else:
            return merged
    except Exception:
        return []


def fixed_window_segments(total_dur_s: float, max_seg_s: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    cur = 0.0
    while cur < total_dur_s:
        end = min(cur + max_seg_s, total_dur_s)
        segs.append((cur, end))
        cur = end
    return segs


def cut_wav_segment(src_wav: str, dst_wav: str, start_s: float, end_s: float) -> None:
    duration = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-i", src_wav,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", dst_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_segments(wav_path: str, segs: List[Tuple[float, float]], tmpdir: str) -> List[Tuple[int, float, float, str]]:
    """Cut segment WAVs to disk; return (idx, start, end, path)."""
    prepared: List[Tuple[int, float, float, str]] = []
    for i, (start_s, end_s) in enumerate(segs, start=1):
        seg_path = f"{tmpdir}/seg_{i:04d}.wav"
        cut_wav_segment(wav_path, seg_path, start_s, end_s)
        prepared.append((i, start_s, end_s, seg_path))
    return prepared
