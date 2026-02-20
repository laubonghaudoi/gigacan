from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from .alignment import align_cues
from .llm import (
    CorrectionCandidate,
    CorrectionModel,
    CorrectionRequest,
    OllamaCorrectionModel,
    VllmCorrectionModel,
)
from .normalize import TextNormalizer, extract_top_terms
from .reporting import YueDriftRow, build_yue_drift_row
from .srt_utils import SubtitleCue, parse_srt, write_srt_cues

ZH_HK_LANGS = ("zh-hk",)
YUE_LANGS = ("yue-hant", "yue")
T = TypeVar("T")


@dataclass(slots=True)
class FileCorrectionStats:
    video_id: str
    year: str
    asr_path: str
    zh_hk_reference: str
    yue_reference: str
    output_path: str
    status: str
    skip_reason: str
    error: str
    total_cues: int
    changed_cues: int
    no_evidence_cues: int
    rejected_cues: int
    avg_confidence: float


@dataclass(slots=True)
class YearCorrectionResult:
    manifest_rows: list[FileCorrectionStats]
    yue_rows: list[YueDriftRow]
    report: dict[str, object]


def _normalise_lang(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _extract_lang_from_filename(path: Path, video_id: str) -> str:
    prefix = f"{video_id}."
    if not path.name.startswith(prefix):
        return ""
    stem_part = path.name[len(prefix) :]
    if "." not in stem_part:
        return ""
    return stem_part.rsplit(".", 1)[0]


def _lang_matches_any(candidate: str, targets: tuple[str, ...]) -> bool:
    candidate_norm = _normalise_lang(candidate)
    for target in targets:
        target_norm = _normalise_lang(target)
        if candidate_norm == target_norm:
            return True
        if candidate_norm.startswith(target_norm + "-"):
            return True
    return False


def find_reference_subtitle(
    *,
    root: Path,
    year: str,
    video_id: str,
    target_langs: tuple[str, ...],
) -> Path | None:
    year_dir = root / year
    if not year_dir.is_dir():
        return None

    candidates: list[Path] = []
    for path in year_dir.glob(f"{video_id}.*"):
        if not path.is_file() or path.suffix.lower() != ".srt":
            continue
        language = _extract_lang_from_filename(path, video_id)
        if not language:
            continue
        if _lang_matches_any(language, target_langs):
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item.name)[0]


def collect_asr_files(asr_root: Path, year: str, *, max_files: int) -> list[Path]:
    year_dir = asr_root / year
    if not year_dir.is_dir():
        raise FileNotFoundError(f"ASR year directory not found: {year_dir}")

    files = sorted(path for path in year_dir.rglob("*.srt") if path.is_file())
    if max_files > 0:
        return files[:max_files]
    return files


def _build_model(
    *,
    backend: str,
    model_name: str,
    ollama_host: str,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    max_new_tokens: int,
    temperature: float,
) -> CorrectionModel:
    resolved_backend = backend.strip().lower()
    if resolved_backend == "vllm":
        return VllmCorrectionModel(
            model=model_name,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    if resolved_backend == "ollama":
        return OllamaCorrectionModel(
            model=model_name,
            host=ollama_host,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported correction backend: {backend}")


def _iter_chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _apply_candidates_to_cues(
    *,
    asr_cues: list[SubtitleCue],
    cue_requests: list[tuple[int, CorrectionRequest]],
    model: CorrectionModel,
    normalizer: TextNormalizer,
    llm_batch_size: int,
) -> tuple[list[SubtitleCue], int, int, float]:
    cue_map: dict[int, CorrectionCandidate] = {}

    requests_only = [request for _, request in cue_requests]
    request_indexes = [cue_index for cue_index, _ in cue_requests]

    for request_chunk, index_chunk in zip(
        _iter_chunks(requests_only, llm_batch_size),
        _iter_chunks(request_indexes, llm_batch_size),
    ):
        try:
            chunk_candidates = model.correct_batch(request_chunk)
        except Exception:
            chunk_candidates = [
                CorrectionCandidate(
                    corrected_text=request.asr_text,
                    change_type="none",
                    confidence=0.0,
                    reason="llm_runtime_error",
                    valid=False,
                    error="llm_runtime_error",
                )
                for request in request_chunk
            ]

        if len(chunk_candidates) != len(request_chunk):
            fixed: list[CorrectionCandidate] = []
            for item_index, request in enumerate(request_chunk):
                if item_index < len(chunk_candidates):
                    fixed.append(chunk_candidates[item_index])
                else:
                    fixed.append(
                        CorrectionCandidate(
                            corrected_text=request.asr_text,
                            change_type="none",
                            confidence=0.0,
                            reason="missing_model_output",
                            valid=False,
                            error="missing_model_output",
                        )
                    )
            chunk_candidates = fixed

        for cue_index, candidate in zip(index_chunk, chunk_candidates):
            cue_map[cue_index] = candidate

    corrected: list[SubtitleCue] = []
    changed = 0
    rejected = 0
    confidence_total = 0.0

    for cue_index, cue in enumerate(asr_cues):
        candidate = cue_map.get(cue_index)
        if candidate is None:
            corrected.append(cue)
            continue

        if not candidate.valid:
            rejected += 1
            corrected.append(cue)
            continue

        cleaned_text = normalizer.clean(candidate.corrected_text)
        if not cleaned_text:
            rejected += 1
            corrected.append(cue)
            continue

        if cleaned_text != cue.text:
            changed += 1
        confidence_total += candidate.confidence
        corrected.append(
            SubtitleCue(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cleaned_text,
            )
        )

    return corrected, changed, rejected, confidence_total


def _safe_parse_srt(path: Path) -> list[SubtitleCue]:
    try:
        return parse_srt(path)
    except Exception:
        return []


def _build_review_sample(
    corrected_video_ids: list[str],
    *,
    size: int,
    seed: int,
) -> list[str]:
    if not corrected_video_ids or size <= 0:
        return []
    rng = random.Random(seed)
    sample_size = min(size, len(corrected_video_ids))
    return sorted(rng.sample(corrected_video_ids, sample_size))


def _write_manifest(path: Path, rows: list[FileCorrectionStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "year",
        "asr_path",
        "zh_hk_reference",
        "yue_reference",
        "output_path",
        "status",
        "skip_reason",
        "error",
        "total_cues",
        "changed_cues",
        "no_evidence_cues",
        "rejected_cues",
        "avg_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["avg_confidence"] = round(float(payload["avg_confidence"]), 6)
            writer.writerow(payload)


def _write_yue_report(path: Path, rows: list[YueDriftRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "year",
        "has_zh_hk_ref",
        "cue_count_asr",
        "cue_count_yue",
        "coverage_time_overlap_ratio",
        "normalized_text_similarity",
        "char_length_ratio_yue_to_asr",
        "top_diff_terms",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow(payload)


def _write_report_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_year_correction(
    *,
    year: str,
    asr_root: Path,
    zh_root: Path,
    yue_root: Path,
    output_root: Path,
    manifest_out: Path,
    report_out: Path,
    yue_report_out: Path,
    backend: str,
    model_name: str,
    ollama_host: str,
    overwrite: bool,
    max_files: int,
    seed: int,
    review_sample_size: int,
    llm_batch_size: int,
    min_alignment_score: float,
    max_alignment_gap_ms: int,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    max_new_tokens: int,
    temperature: float,
    model: CorrectionModel | None = None,
) -> YearCorrectionResult:
    started_at = datetime.now(timezone.utc)
    normalizer = TextNormalizer()

    asr_files = collect_asr_files(asr_root, year, max_files=max_files)

    manifest_rows: list[FileCorrectionStats] = []
    yue_rows: list[YueDriftRow] = []
    corrected_video_ids: list[str] = []

    model_client = model
    totals = {
        "asr_files": len(asr_files),
        "eligible_zh_hk_files": 0,
        "corrected_files": 0,
        "skipped_files": 0,
        "skipped_missing_zh_hk": 0,
        "failed_files": 0,
        "yue_report_files": 0,
        "total_cues": 0,
        "changed_cues": 0,
        "no_evidence_cues": 0,
        "rejected_cues": 0,
    }

    for asr_path in asr_files:
        video_id = asr_path.stem
        zh_ref = find_reference_subtitle(
            root=zh_root,
            year=year,
            video_id=video_id,
            target_langs=ZH_HK_LANGS,
        )
        yue_ref = find_reference_subtitle(
            root=yue_root,
            year=year,
            video_id=video_id,
            target_langs=YUE_LANGS,
        )
        output_path = output_root / year / f"{video_id}.srt"

        asr_cues = _safe_parse_srt(asr_path)
        if yue_ref is not None:
            yue_cues = _safe_parse_srt(yue_ref)
            yue_rows.append(
                build_yue_drift_row(
                    video_id=video_id,
                    year=year,
                    asr_cues=asr_cues,
                    yue_cues=yue_cues,
                    has_zh_hk_ref=zh_ref is not None,
                    normalizer=normalizer,
                )
            )
            totals["yue_report_files"] += 1

        if zh_ref is None:
            totals["skipped_files"] += 1
            totals["skipped_missing_zh_hk"] += 1
            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference="",
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="skipped",
                    skip_reason="missing_zh_hk_reference",
                    error="",
                    total_cues=len(asr_cues),
                    changed_cues=0,
                    no_evidence_cues=len(asr_cues),
                    rejected_cues=0,
                    avg_confidence=0.0,
                )
            )
            continue

        totals["eligible_zh_hk_files"] += 1

        if output_path.exists() and not overwrite:
            totals["skipped_files"] += 1
            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference=str(zh_ref),
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="skipped",
                    skip_reason="output_exists",
                    error="",
                    total_cues=len(asr_cues),
                    changed_cues=0,
                    no_evidence_cues=0,
                    rejected_cues=0,
                    avg_confidence=0.0,
                )
            )
            continue

        zh_cues = _safe_parse_srt(zh_ref)
        if not asr_cues:
            totals["failed_files"] += 1
            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference=str(zh_ref),
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="failed",
                    skip_reason="",
                    error="empty_asr_cues",
                    total_cues=0,
                    changed_cues=0,
                    no_evidence_cues=0,
                    rejected_cues=0,
                    avg_confidence=0.0,
                )
            )
            continue

        if not zh_cues:
            totals["failed_files"] += 1
            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference=str(zh_ref),
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="failed",
                    skip_reason="",
                    error="empty_zh_hk_cues",
                    total_cues=len(asr_cues),
                    changed_cues=0,
                    no_evidence_cues=len(asr_cues),
                    rejected_cues=0,
                    avg_confidence=0.0,
                )
            )
            continue

        if model_client is None:
            model_client = _build_model(
                backend=backend,
                model_name=model_name,
                ollama_host=ollama_host,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        try:
            alignments = align_cues(
                asr_cues,
                zh_cues,
                normalizer=normalizer,
                max_gap_ms=max_alignment_gap_ms,
                min_score=min_alignment_score,
            )

            terminology = extract_top_terms([cue.text for cue in zh_cues], top_k=30)
            cue_requests: list[tuple[int, CorrectionRequest]] = []
            no_evidence = 0
            for alignment in alignments:
                cue = asr_cues[alignment.asr_index]
                if not alignment.evidence_text:
                    no_evidence += 1
                    continue
                cue_requests.append(
                    (
                        alignment.asr_index,
                        CorrectionRequest(
                            asr_text=cue.text,
                            evidence_text=alignment.evidence_text,
                            terminology=terminology,
                        ),
                    )
                )

            corrected_cues, changed_cues, rejected_cues, confidence_total = _apply_candidates_to_cues(
                asr_cues=asr_cues,
                cue_requests=cue_requests,
                model=model_client,
                normalizer=normalizer,
                llm_batch_size=llm_batch_size,
            )
            avg_confidence = confidence_total / len(cue_requests) if cue_requests else 0.0

            write_srt_cues(output_path, corrected_cues)

            totals["corrected_files"] += 1
            totals["total_cues"] += len(asr_cues)
            totals["changed_cues"] += changed_cues
            totals["no_evidence_cues"] += no_evidence
            totals["rejected_cues"] += rejected_cues
            corrected_video_ids.append(video_id)

            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference=str(zh_ref),
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="corrected",
                    skip_reason="",
                    error="",
                    total_cues=len(asr_cues),
                    changed_cues=changed_cues,
                    no_evidence_cues=no_evidence,
                    rejected_cues=rejected_cues,
                    avg_confidence=avg_confidence,
                )
            )
        except Exception as exc:
            totals["failed_files"] += 1
            manifest_rows.append(
                FileCorrectionStats(
                    video_id=video_id,
                    year=year,
                    asr_path=str(asr_path),
                    zh_hk_reference=str(zh_ref),
                    yue_reference=str(yue_ref) if yue_ref is not None else "",
                    output_path=str(output_path),
                    status="failed",
                    skip_reason="",
                    error=str(exc),
                    total_cues=len(asr_cues),
                    changed_cues=0,
                    no_evidence_cues=0,
                    rejected_cues=0,
                    avg_confidence=0.0,
                )
            )

    review_sample_video_ids = _build_review_sample(
        corrected_video_ids,
        size=review_sample_size,
        seed=seed,
    )

    finished_at = datetime.now(timezone.utc)
    report = {
        "year": year,
        "backend": backend,
        "model": model_name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "totals": totals,
        "review_sample_video_ids": review_sample_video_ids,
    }

    _write_manifest(manifest_out, manifest_rows)
    _write_yue_report(yue_report_out, yue_rows)
    _write_report_json(report_out, report)

    return YearCorrectionResult(
        manifest_rows=manifest_rows,
        yue_rows=yue_rows,
        report=report,
    )
