from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_year_correction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Correct Qwen Cantonese subtitles using zh-HK references only, and "
            "generate yue-vs-ASR drift report."
        )
    )
    parser.add_argument("--year", required=True, help='Target year directory (e.g., "2025").')
    parser.add_argument("--asr-root", default="transcriptions", help='ASR subtitle root. Default: "transcriptions".')
    parser.add_argument("--ref-zh-root", default="zh-hk", help='zh-HK reference subtitle root. Default: "zh-hk".')
    parser.add_argument("--ref-yue-root", default="yue", help='yue subtitle root for reporting only. Default: "yue".')
    parser.add_argument("--output-root", default="corrected_transcriptions", help='Corrected subtitle output root. Default: "corrected_transcriptions".')
    parser.add_argument("--manifest-out", default="", help="CSV manifest output path.")
    parser.add_argument("--report-out", default="", help="JSON summary report output path.")
    parser.add_argument("--yue-report-out", default="", help="CSV yue drift report output path.")
    parser.add_argument(
        "--model",
        default="google/translategemma-27b-it",
        help="Model id/tag. For vLLM: HF model id. For Ollama: local tag (e.g., gemma3:27b).",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "ollama"],
        help='Correction backend. Default: "vllm".',
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help='Ollama host URL when --backend ollama. Default: "http://127.0.0.1:11434".',
    )
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N ASR files (0 = all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing corrected outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic review sampling.")
    parser.add_argument("--review-sample-size", type=int, default=120, help="Manual review sample size in report. Default: 120.")
    parser.add_argument("--llm-batch-size", type=int, default=8, help="Number of cues per correction batch call. Default: 8.")
    parser.add_argument("--min-alignment-score", type=float, default=0.22, help="Minimum cue alignment score for using zh-HK evidence.")
    parser.add_argument("--max-alignment-gap-ms", type=int, default=4000, help="Maximum cue gap (ms) for alignment candidates.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization target.")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Maximum generated tokens per cue correction.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.max_files < 0:
        parser.error("--max-files must be >= 0.")
    if ns.review_sample_size < 0:
        parser.error("--review-sample-size must be >= 0.")
    if ns.llm_batch_size < 1:
        parser.error("--llm-batch-size must be >= 1.")
    if not (0.0 <= ns.temperature <= 2.0):
        parser.error("--temperature must be in [0, 2].")
    if ns.min_alignment_score <= 0.0 or ns.min_alignment_score > 1.0:
        parser.error("--min-alignment-score must be in (0, 1].")
    if ns.max_alignment_gap_ms < 0:
        parser.error("--max-alignment-gap-ms must be >= 0.")
    if ns.vllm_tensor_parallel_size < 1:
        parser.error("--vllm-tensor-parallel-size must be >= 1.")
    if ns.vllm_gpu_memory_utilization <= 0.0 or ns.vllm_gpu_memory_utilization > 1.0:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1].")
    if ns.max_new_tokens < 16:
        parser.error("--max-new-tokens must be >= 16.")

    return ns


def _resolve_default_output_paths(
    *,
    year: str,
    manifest_out: str,
    report_out: str,
    yue_report_out: str,
) -> tuple[Path, Path, Path]:
    manifest = Path(manifest_out) if manifest_out else Path("logs") / f"correction_manifest_{year}.csv"
    report = Path(report_out) if report_out else Path("logs") / f"correction_report_{year}.json"
    yue_report = Path(yue_report_out) if yue_report_out else Path("logs") / f"yue_drift_report_{year}.csv"
    return manifest, report, yue_report


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    manifest_out, report_out, yue_report_out = _resolve_default_output_paths(
        year=str(ns.year),
        manifest_out=ns.manifest_out,
        report_out=ns.report_out,
        yue_report_out=ns.yue_report_out,
    )

    result = run_year_correction(
        year=str(ns.year),
        asr_root=Path(ns.asr_root),
        zh_root=Path(ns.ref_zh_root),
        yue_root=Path(ns.ref_yue_root),
        output_root=Path(ns.output_root),
        manifest_out=manifest_out,
        report_out=report_out,
        yue_report_out=yue_report_out,
        backend=ns.backend,
        model_name=ns.model,
        ollama_host=str(ns.ollama_host),
        overwrite=bool(ns.overwrite),
        max_files=int(ns.max_files),
        seed=int(ns.seed),
        review_sample_size=int(ns.review_sample_size),
        llm_batch_size=int(ns.llm_batch_size),
        min_alignment_score=float(ns.min_alignment_score),
        max_alignment_gap_ms=int(ns.max_alignment_gap_ms),
        vllm_gpu_memory_utilization=float(ns.vllm_gpu_memory_utilization),
        vllm_tensor_parallel_size=int(ns.vllm_tensor_parallel_size),
        max_new_tokens=int(ns.max_new_tokens),
        temperature=float(ns.temperature),
    )

    totals = result.report.get("totals", {})
    print("Correction pass completed.")
    print(f"Manifest: {manifest_out}")
    print(f"Report: {report_out}")
    print(f"Yue drift report: {yue_report_out}")
    print(f"Totals: {totals}")


if __name__ == "__main__":
    main()
