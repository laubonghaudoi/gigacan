from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .batch import BatchJob
from .config import TranscribeConfig

WORKER_START_TIMEOUT_SECONDS = 600.0


def _send_json(writer: Any, payload: dict) -> None:
    writer.write(json.dumps(payload, ensure_ascii=True) + "\n")
    writer.flush()


def _read_json(reader: Any) -> dict:
    line = reader.readline()
    if not line:
        raise RuntimeError("Worker connection closed unexpectedly.")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid worker payload: {value}")
    return value


def _worker_command(config: TranscribeConfig, socket_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "gigacan.qwen_srt.worker_daemon",
        "--socket",
        str(socket_path),
        "--device",
        config.device,
        "--segment-batch-size",
        str(config.segment_batch_size),
        "--min-segment-ms",
        str(config.min_segment_ms),
        "--vad-max-segment-ms",
        str(config.vad_max_segment_ms),
        "--vad-max-end-silence-ms",
        str(config.vad_max_end_silence_ms),
        "--merge-target-segment-ms",
        str(config.merge_target_segment_ms),
        "--merge-max-segment-ms",
        str(config.merge_max_segment_ms),
        "--merge-max-gap-ms",
        str(config.merge_max_gap_ms),
        "--prep-workers",
        str(config.prep_workers),
        "--vad-workers",
        str(config.vad_workers),
        "--vad-device",
        config.vad_device,
        "--asr-prefetch-batches",
        str(config.asr_prefetch_batches),
        "--asr-backend",
        config.asr_backend,
        "--asr-model",
        config.asr_model,
        "--asr-model-hub",
        config.asr_model_hub,
        "--asr-language",
        config.asr_language,
        "--vad-cache-dir",
        str(config.vad_cache_dir),
    ]
    if not config.asr_use_itn:
        cmd.append("--no-asr-use-itn")
    if not config.use_vad_cache:
        cmd.append("--no-vad-cache")
    return cmd


def _runtime_signature(config: TranscribeConfig) -> dict[str, Any]:
    return {
        "device": config.device,
        "segment_batch_size": config.segment_batch_size,
        "min_segment_ms": config.min_segment_ms,
        "vad_max_segment_ms": config.vad_max_segment_ms,
        "vad_max_end_silence_ms": config.vad_max_end_silence_ms,
        "merge_target_segment_ms": config.merge_target_segment_ms,
        "merge_max_segment_ms": config.merge_max_segment_ms,
        "merge_max_gap_ms": config.merge_max_gap_ms,
        "prep_workers": config.prep_workers,
        "vad_workers": config.vad_workers,
        "vad_device": config.vad_device,
        "asr_prefetch_batches": config.asr_prefetch_batches,
        "asr_backend": config.asr_backend,
        "asr_model": config.asr_model,
        "asr_model_hub": config.asr_model_hub,
        "asr_language": config.asr_language,
        "asr_use_itn": config.asr_use_itn,
        "vad_cache_dir": str(config.vad_cache_dir),
        "use_vad_cache": config.use_vad_cache,
    }


def _connect(socket_path: Path) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    return client


def _worker_log_path(socket_path: Path) -> Path:
    return socket_path.with_suffix(".worker.log")


def ping_worker(socket_path: Path) -> dict[str, Any] | None:
    try:
        with _connect(socket_path) as client:
            reader = client.makefile("r", encoding="utf-8")
            writer = client.makefile("w", encoding="utf-8")
            _send_json(writer, {"action": "ping"})
            response = _read_json(reader)
            if response.get("event") != "pong":
                return None
            return response
    except Exception:
        return None


def shutdown_worker(socket_path: Path) -> bool:
    if not socket_path.exists():
        return False
    try:
        with _connect(socket_path) as client:
            reader = client.makefile("r", encoding="utf-8")
            writer = client.makefile("w", encoding="utf-8")
            _send_json(writer, {"action": "shutdown"})
            response = _read_json(reader)
            return response.get("event") == "shutting_down"
    except Exception:
        return False


def ensure_worker_running(config: TranscribeConfig, socket_path: Path) -> None:
    expected_signature = _runtime_signature(config)
    pong = ping_worker(socket_path)
    if pong is not None and pong.get("signature") == expected_signature:
        return
    if pong is not None:
        stopped = shutdown_worker(socket_path)
        if not stopped:
            raise RuntimeError(
                "A worker is already running with a different config, "
                f"but it could not be stopped at {socket_path}."
            )
        time.sleep(0.5)

    if socket_path.exists():
        try:
            socket_path.unlink()
        except Exception:
            pass

    socket_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _worker_command(config, socket_path)
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_dir
        if not existing_pythonpath
        else f"{src_dir}{os.pathsep}{existing_pythonpath}"
    )
    worker_log = _worker_log_path(socket_path)
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with worker_log.open("ab") as log_fh:
        process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    deadline = time.time() + WORKER_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        pong = ping_worker(socket_path)
        if pong is not None and pong.get("signature") == expected_signature:
            return
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Worker process exited during startup with code {exit_code}."
            )
        time.sleep(1.0)

    raise TimeoutError(
        f"Timed out waiting for worker startup at {socket_path} after "
        f"{int(WORKER_START_TIMEOUT_SECONDS)} seconds."
    )


def transcribe_batch_via_worker(
    *,
    config: TranscribeConfig,
    jobs: Sequence[BatchJob],
    socket_path: Path,
    max_active_files: int,
    queue_multiplier: int,
    preload_files: int,
    max_decoded_gib: float,
    continue_on_error: bool,
    on_file_done: Callable[[Path], None] | None = None,
    on_file_failed: Callable[[Path, str], None] | None = None,
) -> tuple[int, list[tuple[Path, str]]]:
    ensure_worker_running(config, socket_path)

    request = {
        "action": "batch",
        "jobs": [
            {
                "audio": str(job.audio),
                "output_srt": str(job.output_srt),
            }
            for job in jobs
        ],
        "max_active_files": int(max_active_files),
        "queue_multiplier": int(queue_multiplier),
        "preload_files": int(preload_files),
        "max_decoded_gib": float(max_decoded_gib),
        "continue_on_error": bool(continue_on_error),
    }

    with _connect(socket_path) as client:
        reader = client.makefile("r", encoding="utf-8")
        writer = client.makefile("w", encoding="utf-8")
        _send_json(writer, request)

        while True:
            event = _read_json(reader)
            event_type = event.get("event")
            if event_type == "file_done":
                audio_str = str(event.get("audio", ""))
                if on_file_done is not None and audio_str:
                    on_file_done(Path(audio_str))
                continue

            if event_type == "file_failed":
                audio_str = str(event.get("audio", ""))
                message = str(event.get("message", "Unknown worker error"))
                if on_file_failed is not None and audio_str:
                    on_file_failed(Path(audio_str), message)
                continue

            if event_type == "result":
                completed = int(event.get("completed", 0))
                raw_failures = event.get("failures", [])
                failures: list[tuple[Path, str]] = []
                if isinstance(raw_failures, list):
                    for item in raw_failures:
                        if isinstance(item, list | tuple) and len(item) == 2:
                            failures.append((Path(str(item[0])), str(item[1])))
                return completed, failures

            if event_type == "error":
                message = str(event.get("message", "Unknown worker error"))
                raise RuntimeError(message)

            raise RuntimeError(f"Unexpected worker event: {event}")
