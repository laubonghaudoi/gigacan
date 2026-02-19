from __future__ import annotations

import argparse
import json
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .batch import BatchJob
from .config import TranscribeConfig
from .pipeline import prepare_transcriber, transcribe_batch_to_srt_superbatched


def _send_json(writer: Any, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, ensure_ascii=True) + "\n")
    writer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Qwen SRT worker daemon")
    parser.add_argument("--socket", required=True, help="UNIX socket path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segment-batch-size", type=int, default=0)
    parser.add_argument("--min-segment-ms", type=int, default=300)
    parser.add_argument("--vad-max-segment-ms", type=int, default=20000)
    parser.add_argument("--merge-target-segment-ms", type=int, default=4000)
    parser.add_argument("--merge-max-segment-ms", type=int, default=12000)
    parser.add_argument("--merge-max-gap-ms", type=int, default=250)
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument("--vad-workers", type=int, default=1)
    parser.add_argument(
        "--vad-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--asr-prefetch-batches", type=int, default=2)
    parser.add_argument("--qwen-src-dir", default=".cache/Qwen3-ASR-src")
    parser.add_argument(
        "--qwen-repo-url",
        default="https://github.com/QwenLM/Qwen3-ASR",
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument(
        "--asr-backend",
        default="vllm",
        choices=["vllm", "transformers"],
    )
    parser.add_argument("--qwen-language", default="Cantonese")
    parser.add_argument("--qwen-context", default="")
    parser.add_argument("--use-prompt", action="store_true")
    parser.add_argument(
        "--qwen-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--vad-cache-dir", default=".cache/qwen_srt_vad")
    parser.add_argument("--no-vad-cache", action="store_true")
    return parser


def _build_runtime_config(ns: argparse.Namespace) -> TranscribeConfig:
    return TranscribeConfig(
        audio=Path("__worker_placeholder__.wav"),
        output_srt=Path("__worker_placeholder__.srt"),
        device=ns.device,
        segment_batch_size=ns.segment_batch_size,
        min_segment_ms=ns.min_segment_ms,
        vad_max_segment_ms=ns.vad_max_segment_ms,
        merge_target_segment_ms=ns.merge_target_segment_ms,
        merge_max_segment_ms=ns.merge_max_segment_ms,
        merge_max_gap_ms=ns.merge_max_gap_ms,
        prep_workers=ns.prep_workers,
        vad_workers=ns.vad_workers,
        vad_device=ns.vad_device,
        asr_prefetch_batches=ns.asr_prefetch_batches,
        qwen_src_dir=Path(ns.qwen_src_dir),
        qwen_repo_url=ns.qwen_repo_url,
        qwen_model=ns.qwen_model,
        asr_backend=ns.asr_backend,
        qwen_language=ns.qwen_language,
        qwen_context=ns.qwen_context,
        use_prompt=ns.use_prompt,
        qwen_dtype=ns.qwen_dtype,
        vllm_gpu_memory_utilization=ns.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=ns.vllm_tensor_parallel_size,
        qwen_max_new_tokens=ns.qwen_max_new_tokens,
        vad_cache_dir=Path(ns.vad_cache_dir),
        use_vad_cache=not ns.no_vad_cache,
    )


def _parse_jobs(payload: dict[str, Any]) -> list[BatchJob]:
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("Invalid request: jobs must be a list")

    jobs: list[BatchJob] = []
    for item in raw_jobs:
        if not isinstance(item, dict):
            raise ValueError("Invalid request: each job must be an object")
        audio = item.get("audio")
        output_srt = item.get("output_srt")
        if not isinstance(audio, str) or not isinstance(output_srt, str):
            raise ValueError("Invalid request: job audio/output_srt must be strings")
        jobs.append(BatchJob(audio=Path(audio), output_srt=Path(output_srt)))
    return jobs


def run_server(namespace: argparse.Namespace) -> None:
    socket_path = Path(namespace.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    config = _build_runtime_config(namespace)
    runtime = prepare_transcriber(config)
    runtime_signature = {
        "device": config.device,
        "segment_batch_size": config.segment_batch_size,
        "min_segment_ms": config.min_segment_ms,
        "vad_max_segment_ms": config.vad_max_segment_ms,
        "merge_target_segment_ms": config.merge_target_segment_ms,
        "merge_max_segment_ms": config.merge_max_segment_ms,
        "merge_max_gap_ms": config.merge_max_gap_ms,
        "prep_workers": config.prep_workers,
        "vad_workers": config.vad_workers,
        "vad_device": config.vad_device,
        "asr_prefetch_batches": config.asr_prefetch_batches,
        "qwen_src_dir": str(config.qwen_src_dir),
        "qwen_repo_url": config.qwen_repo_url,
        "qwen_model": config.qwen_model,
        "asr_backend": config.asr_backend,
        "qwen_language": config.qwen_language,
        "qwen_context": config.qwen_context,
        "use_prompt": config.use_prompt,
        "qwen_dtype": config.qwen_dtype,
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
        "qwen_max_new_tokens": config.qwen_max_new_tokens,
        "vad_cache_dir": str(config.vad_cache_dir),
        "use_vad_cache": config.use_vad_cache,
    }

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)

        while True:
            conn, _ = server.accept()
            with conn:
                reader = conn.makefile("r", encoding="utf-8")
                writer = conn.makefile("w", encoding="utf-8")

                line = reader.readline()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("Request must be a JSON object")

                    action = payload.get("action")
                    if action == "ping":
                        _send_json(
                            writer,
                            {"event": "pong", "signature": runtime_signature},
                        )
                        continue

                    if action == "shutdown":
                        _send_json(writer, {"event": "shutting_down"})
                        return

                    if action != "batch":
                        raise ValueError(f"Unsupported action: {action}")

                    jobs = _parse_jobs(payload)
                    max_active_files = int(payload.get("max_active_files", 0))
                    queue_multiplier = int(payload.get("queue_multiplier", 4))
                    preload_files = int(payload.get("preload_files", 0))
                    max_decoded_gib = float(payload.get("max_decoded_gib", 0.0))
                    continue_on_error = bool(payload.get("continue_on_error", False))

                    def on_done(audio_path: Path) -> None:
                        _send_json(
                            writer,
                            {
                                "event": "file_done",
                                "audio": str(audio_path),
                            },
                        )

                    def on_failed(audio_path: Path, message: str) -> None:
                        _send_json(
                            writer,
                            {
                                "event": "file_failed",
                                "audio": str(audio_path),
                                "message": message,
                            },
                        )

                    completed, failures = transcribe_batch_to_srt_superbatched(
                        runtime,
                        jobs,
                        max_active_files=max_active_files,
                        queue_multiplier=queue_multiplier,
                        preload_files=preload_files,
                        max_decoded_gib=max_decoded_gib,
                        continue_on_error=continue_on_error,
                        on_file_done=on_done,
                        on_file_failed=on_failed,
                    )
                    _send_json(
                        writer,
                        {
                            "event": "result",
                            "completed": completed,
                            "failures": [
                                [str(audio_path), message]
                                for audio_path, message in failures
                            ],
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    _send_json(
                        writer,
                        {
                            "event": "error",
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    )
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    run_server(namespace)


if __name__ == "__main__":
    main()
