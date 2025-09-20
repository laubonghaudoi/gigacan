"""Batch VTT generation CLI matching the former Jupyter notebook workflow.

Reads a playlist CSV, downloads audio (if needed), transcribes with Qwen-ASR,
and writes individual WebVTT files while updating the CSV completion status.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# Ensure src/ is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gigacan import segmenter, transcriber  # noqa: E402
from gigacan.corrector import RuleBasedCorrector  # noqa: E402

PROMPT_TEMPLATE = """以下係香港立法會會議錄音嘅一段節選，佢嘅標題係{title}，會議介紹如下：
---
{description}
---

請將呢段錄音轉寫成標準粵文，唔好寫普通話，要求區分「咁噉」「係系喺」，除非固定譯名用「俾」之外規定都用「畀」。語氣詞要用「呢」，除非係 le4 先至寫「咧」。
"""


@dataclass
class PlaylistRow:
    index: int
    data: Dict[str, str]


class StreamLogger:
    """File-like helper that forwards stdout/stderr to a callback."""

    def __init__(self, callback: Optional[Callable[[str], None]]) -> None:
        self._callback = callback
        self._buffer: str = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        if self._callback is None:
            return len(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._callback(stripped)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._callback is None:
            return
        stripped = self._buffer.strip()
        if stripped:
            self._callback(stripped)
        self._buffer = ""


def escape_braces(value: str) -> str:
    return (value or "").replace("{", "{{").replace("}", "}}")


def build_system_prompt(template: str, title: str, description: str) -> str:
    return template.format(
        title=escape_braces(title or ""),
        description=escape_braces(description or ""),
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_playlist_rows(csv_path: Path) -> Tuple[List[PlaylistRow], List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [PlaylistRow(index=i, data=dict(row)) for i, row in enumerate(reader)]
    if "completed" not in fieldnames:
        fieldnames.append("completed")
        for row in rows:
            row.data["completed"] = "FALSE"
    else:
        for row in rows:
            completed_raw = str(row.data.get("completed", "")).strip().upper()
            row.data["completed"] = "TRUE" if completed_raw in {"TRUE", "1", "YES"} else "FALSE"
    return rows, fieldnames


def write_playlist_rows(csv_path: Path, fieldnames: List[str], rows: Iterable[PlaylistRow]) -> None:
    ensure_directory(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.data)


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _fmt_rate(speed: Optional[float]) -> str:
    if not speed:
        return "?"
    units = ["B", "KB", "MB", "GB"]
    rate = float(speed)
    idx = 0
    while rate >= 1024 and idx < len(units) - 1:
        rate /= 1024
        idx += 1
    return f"{rate:.1f}{units[idx]}/s"


def download_audio(
    url: str,
    *,
    output_dir: Path,
    audio_format: str,
    sample_rate: Optional[int],
    sleep_interval: Optional[float],
    max_sleep_interval: Optional[float],
    skip_existing: bool,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, dict]:
    ensure_directory(output_dir)
    if log_callback:
        log_callback(f"[download] Resolving {url}")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": audio_format}],
        "overwrites": not skip_existing,
    }
    if sample_rate:
        ydl_opts["postprocessor_args"] = ["-ar", str(sample_rate)]
    if sleep_interval and sleep_interval > 0:
        ydl_opts["sleep_interval"] = float(sleep_interval)
    if max_sleep_interval and max_sleep_interval > 0:
        if not sleep_interval or max_sleep_interval >= sleep_interval:
            ydl_opts["max_sleep_interval"] = float(max_sleep_interval)

    last_hook_emit = {"time": 0.0}
    video_label = {"value": url}

    def hook(status_dict: dict) -> None:
        if log_callback is None:
            return
        status = status_dict.get("status")
        if status == "downloading":
            now = time.monotonic()
            if now - last_hook_emit["time"] < 0.5:
                return
            last_hook_emit["time"] = now
            percent = status_dict.get("_percent_str", "").strip()
            eta = _fmt_seconds(status_dict.get("eta"))
            rate = _fmt_rate(status_dict.get("speed"))
            log_callback(
                f"[download] {video_label['value']} {percent or '?'} at {rate} (ETA {eta})"
            )
        elif status == "finished":
            log_callback("[download] Download finished, running post-processing")

    ydl_opts["progress_hooks"] = [hook]

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id")
        video_label["value"] = video_id or url
        target_path = output_dir / f"{video_label['value']}.{audio_format}"
        if skip_existing and target_path.exists():
            if log_callback:
                log_callback(f"[download] Reusing existing audio at {target_path}")
            return target_path, info
        if log_callback:
            log_callback(f"[download] Fetching audio for {video_label['value']}")
        info = ydl.extract_info(url, download=True)
        if not target_path.exists():
            matches = sorted(output_dir.glob(f"{video_label['value']}.*"))
            if matches:
                target_path = matches[0]
            else:
                raise FileNotFoundError(f"Expected audio file {target_path} was not created.")
        if log_callback:
            log_callback(f"[download] Saved audio to {target_path}")
        return target_path, info


def transcribe_media(
    media_path: Path,
    vtt_path: Path,
    *,
    language: str,
    enable_itn: bool,
    concurrency: int,
    max_rpm: int,
    max_retries: int,
    backoff_base: float,
    max_seg_seconds: float,
    vad_merge_ms: int,
    min_speech_ms: int,
    system_prompt: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    ensure_directory(vtt_path.parent)
    if log_callback:
        log_callback(f"[transcribe] Preparing audio from {media_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio_16k_mono.wav"
        try:
            segmenter.extract_mono_wav(str(media_path), str(wav_path), sr=16000)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg failed to extract audio: {exc}") from exc

        total_duration = segmenter.ffprobe_duration_seconds(str(wav_path))
        if total_duration <= 0:
            raise RuntimeError("Could not determine media duration.")
        if log_callback:
            log_callback(f"[transcribe] Source duration {total_duration:.1f}s")

        merge_ms = max(0, int(vad_merge_ms))
        min_speech = max(0, int(min_speech_ms))
        max_seg = float(max_seg_seconds)
        segs = segmenter.try_silero_vad_segments(
            str(wav_path),
            max_seg_s=max_seg,
            merge_gap_ms=merge_ms,
            min_speech_ms=min_speech,
        )
        if segs:
            if log_callback:
                log_callback(f"[transcribe] VAD produced {len(segs)} segments")
        else:
            fallback_window = max(max_seg, 30.0) if max_seg > 0 else 30.0
            segs = segmenter.fixed_window_segments(total_duration, fallback_window)
            if log_callback:
                log_callback(
                    f"[transcribe] VAD failed; falling back to fixed windows ({len(segs)} segments)"
                )

        prepared = segmenter.prepare_segments(str(wav_path), segs, tmpdir)
        if log_callback:
            log_callback(
                f"[transcribe] Prepared {len(prepared)} segment files (concurrency={concurrency})"
            )

        logger = StreamLogger(log_callback)
        with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
            entries = transcriber.transcribe_segments(
                prepared,
                language=language or "zh",
                enable_itn=enable_itn,
                concurrency=max(1, int(concurrency)),
                max_rpm=max(1, int(max_rpm)),
                max_retries=max(0, int(max_retries)),
                backoff_base=float(backoff_base),
                system_prompt=system_prompt,
            )
        logger.flush()

        entries = RuleBasedCorrector().correct_entries(entries)
        if log_callback:
            log_callback(f"[transcribe] Writing VTT to {vtt_path}")
        transcriber.write_webvtt(str(vtt_path), entries)
        if log_callback:
            log_callback("[transcribe] Completed transcription")


def select_candidates(
    rows: List[PlaylistRow],
    *,
    reprocess_completed: bool,
    limit: int,
) -> List[PlaylistRow]:
    candidates: List[PlaylistRow] = []
    for row in rows:
        if not reprocess_completed and row.data.get("completed") == "TRUE":
            continue
        candidates.append(row)
        if limit and len(candidates) >= limit:
            break
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcript to WebVTT using Qwen-ASR")
    parser.add_argument("--csv", default="legco.csv", help="Path to playlist CSV (default: legco.csv)")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows (0 means all)")
    parser.add_argument("--audio-dir", default="download", help="Directory for downloaded audio")
    parser.add_argument("--vtt-dir", default="vtt", help="Directory for generated WebVTT files")
    parser.add_argument("--audio-format", default="opus", choices=["opus", "mp3", "wav", "m4a"], help="Audio format")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for extracted audio")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between yt-dlp requests")
    parser.add_argument("--sleep-max", type=float, default=0.0, help="Optional random upper bound for sleep interval")
    parser.add_argument("--reuse-audio", action="store_true", default=True, help="Reuse existing audio files if present")
    parser.add_argument("--no-reuse-audio", dest="reuse_audio", action="store_false")
    parser.add_argument("--reuse-vtt", action="store_true", default=True, help="Skip rows when VTT already exists")
    parser.add_argument("--no-reuse-vtt", dest="reuse_vtt", action="store_false")
    parser.add_argument("--reprocess-completed", action="store_true", help="Reprocess rows marked completed")
    parser.add_argument("--language", default="zh", help="ASR language code (use 'auto' to detect)")
    parser.add_argument("--disable-itn", action="store_true", help="Disable inverse text normalization")
    parser.add_argument("--max-seg-seconds", type=float, default=0.0, help="Maximum segment length (0 disables extra splitting)")
    parser.add_argument("--vad-merge-ms", type=int, default=100, help="Merge pauses shorter than this many ms")
    parser.add_argument("--min-speech-ms", type=int, default=200, help="Minimum speech duration for VAD")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent transcription workers")
    parser.add_argument("--max-rpm", type=int, default=60, help="ASR requests per minute")
    parser.add_argument("--retries", type=int, default=3, help="Per-segment retry attempts for transient errors")
    parser.add_argument("--backoff", type=float, default=0.8, help="Backoff base seconds for retries")
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="System prompt template with {title} and {description} placeholders",
    )
    parser.add_argument("--csv-backup", action="store_true", help="Write CSV updates only after successful completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    download_dir = Path(args.audio_dir).expanduser().resolve()
    vtt_dir = Path(args.vtt_dir).expanduser().resolve()
    sample_rate = args.sample_rate if args.sample_rate > 0 else None

    def log_line(message: str) -> None:
        sys.__stdout__.write(f"{message}\n")
        sys.__stdout__.flush()

    try:
        rows, fieldnames = load_playlist_rows(csv_path)
    except Exception as exc:
        log_line(f"[error] Failed to read CSV: {exc}")
        log_line(traceback.format_exc())
        raise SystemExit(1)

    candidates = select_candidates(
        rows,
        reprocess_completed=bool(args.reprocess_completed),
        limit=max(0, int(args.limit)),
    )
    if not candidates:
        log_line("[info] Nothing to process.")
        return

    log_line(f"[info] Processing {len(candidates)} row(s) from {csv_path}")

    try:
        segmenter.check_ffmpeg()
    except Exception as exc:
        log_line(f"[error] FFmpeg check failed: {exc}")
        raise SystemExit(1)

    if args.csv_backup:
        backup_path = csv_path.with_suffix(".bak")
        backup_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        log_line(f"[info] Wrote CSV backup to {backup_path}")

    failures = 0
    for position, row in enumerate(candidates, start=1):
        data = row.data
        title = data.get("title") or ""
        description = data.get("description") or ""
        url = (data.get("url") or "").strip()
        row_label = title or url or f"Row {row.index + 1}"
        log_line(f"[row {row.index + 1}] Started {row_label} ({position}/{len(candidates)})")

        if not url:
            failures += 1
            log_line(f"[skip] Row {row.index + 1} has no URL.")
            data["completed"] = "FALSE"
            write_playlist_rows(csv_path, fieldnames, rows)
            continue

        try:
            audio_path, info = download_audio(
                url,
                output_dir=download_dir,
                audio_format=args.audio_format,
                sample_rate=sample_rate,
                sleep_interval=args.sleep,
                max_sleep_interval=args.sleep_max,
                skip_existing=bool(args.reuse_audio),
                log_callback=log_line,
            )
        except DownloadError as exc:
            failures += 1
            log_line(f"[error] yt-dlp failed for {url}: {exc}")
            log_line(traceback.format_exc())
            data["completed"] = "FALSE"
            write_playlist_rows(csv_path, fieldnames, rows)
            continue
        except Exception as exc:
            failures += 1
            log_line(f"[error] Download error for {url}: {exc}")
            log_line(traceback.format_exc())
            data["completed"] = "FALSE"
            write_playlist_rows(csv_path, fieldnames, rows)
            continue

        video_id = info.get("id") or audio_path.stem
        vtt_path = vtt_dir / f"{video_id}.vtt"
        system_prompt = build_system_prompt(args.prompt_template, title, description)

        if args.reuse_vtt and vtt_path.exists():
            data["completed"] = "TRUE"
            log_line(f"[skip] Reusing existing VTT at {vtt_path}")
            write_playlist_rows(csv_path, fieldnames, rows)
            continue

        try:
            transcribe_media(
                audio_path,
                vtt_path,
                language=args.language,
                enable_itn=not bool(args.disable_itn),
                concurrency=int(args.concurrency),
                max_rpm=int(args.max_rpm),
                max_retries=int(args.retries),
                backoff_base=float(args.backoff),
                max_seg_seconds=float(args.max_seg_seconds),
                vad_merge_ms=int(args.vad_merge_ms),
                min_speech_ms=int(args.min_speech_ms),
                system_prompt=system_prompt,
                log_callback=log_line,
            )
        except Exception as exc:
            failures += 1
            data["completed"] = "FALSE"
            log_line(f"[error] Transcription failed for {video_id}: {exc}")
            log_line(traceback.format_exc())
        else:
            data["completed"] = "TRUE"
            log_line(f"[done] Saved VTT to {vtt_path}")
        finally:
            write_playlist_rows(csv_path, fieldnames, rows)

    if failures:
        log_line(f"[warn] Finished with {failures} failure(s).")
        raise SystemExit(1)

    log_line("[info] All rows completed.")


if __name__ == "__main__":
    main()
