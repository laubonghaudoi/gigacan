from __future__ import annotations

import csv
import json
from pathlib import Path

from gigacan.subtitle_correction.llm import CorrectionCandidate, CorrectionRequest
from gigacan.subtitle_correction.pipeline import (
    find_reference_subtitle,
    run_year_correction,
)
from gigacan.subtitle_correction.srt_utils import parse_srt_content


class DummyCorrectionModel:
    def correct_batch(self, requests: list[CorrectionRequest]) -> list[CorrectionCandidate]:
        out: list[CorrectionCandidate] = []
        for request in requests:
            corrected = request.asr_text
            if request.evidence_text:
                corrected = request.evidence_text
            out.append(
                CorrectionCandidate(
                    corrected_text=corrected,
                    change_type="name_fix",
                    confidence=0.95,
                    reason="test",
                    valid=True,
                )
            )
        return out


class NeverCallModel:
    def correct_batch(self, requests: list[CorrectionRequest]) -> list[CorrectionCandidate]:
        raise AssertionError(f"Model should not be called, got {len(requests)} requests")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_parse_srt_content_handles_multiline_blocks() -> None:
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "第一行\n"
        "第二行\n\n"
        "2\n"
        "00:00:03,500 --> 00:00:05,000\n"
        "第三行\n"
    )

    cues = parse_srt_content(content)
    assert len(cues) == 2
    assert cues[0].text == "第一行\n第二行"
    assert cues[1].text == "第三行"


def test_find_reference_subtitle_prefers_target_language(tmp_path: Path) -> None:
    zh_root = tmp_path / "zh-hk"
    yue_root = tmp_path / "yue"

    _write(zh_root / "2025" / "abc.zh-HK.srt", "")
    _write(yue_root / "2025" / "abc.yue.srt", "")

    zh = find_reference_subtitle(
        root=zh_root,
        year="2025",
        video_id="abc",
        target_langs=("zh-hk",),
    )
    yue = find_reference_subtitle(
        root=yue_root,
        year="2025",
        video_id="abc",
        target_langs=("yue",),
    )

    assert zh == zh_root / "2025" / "abc.zh-HK.srt"
    assert yue == yue_root / "2025" / "abc.yue.srt"


def test_run_year_correction_skips_missing_zh_but_writes_yue_report(tmp_path: Path) -> None:
    asr_root = tmp_path / "transcriptions"
    zh_root = tmp_path / "zh-hk"
    yue_root = tmp_path / "yue"
    output_root = tmp_path / "corrected_transcriptions"

    _write(
        asr_root / "2025" / "vid1.srt",
        "1\n00:00:01,000 --> 00:00:02,000\n你好\n",
    )
    _write(
        yue_root / "2025" / "vid1.yue.srt",
        "1\n00:00:01,000 --> 00:00:02,000\n大家好\n",
    )

    manifest_out = tmp_path / "logs" / "manifest.csv"
    report_out = tmp_path / "logs" / "report.json"
    yue_report_out = tmp_path / "logs" / "yue_report.csv"

    result = run_year_correction(
        year="2025",
        asr_root=asr_root,
        zh_root=zh_root,
        yue_root=yue_root,
        output_root=output_root,
        manifest_out=manifest_out,
        report_out=report_out,
        yue_report_out=yue_report_out,
        backend="vllm",
        model_name="dummy",
        ollama_host="http://127.0.0.1:11434",
        overwrite=False,
        max_files=0,
        seed=42,
        review_sample_size=10,
        llm_batch_size=4,
        min_alignment_score=0.2,
        max_alignment_gap_ms=4000,
        vllm_gpu_memory_utilization=0.9,
        vllm_tensor_parallel_size=1,
        max_new_tokens=64,
        temperature=0.0,
        model=NeverCallModel(),
    )

    assert len(result.manifest_rows) == 1
    assert result.manifest_rows[0].status == "skipped"
    assert result.manifest_rows[0].skip_reason == "missing_zh_hk_reference"
    assert not (output_root / "2025" / "vid1.srt").exists()

    with yue_report_out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["video_id"] == "vid1"

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["totals"]["skipped_missing_zh_hk"] == 1


def test_run_year_correction_uses_zh_reference_for_correction(tmp_path: Path) -> None:
    asr_root = tmp_path / "transcriptions"
    zh_root = tmp_path / "zh-hk"
    yue_root = tmp_path / "yue"
    output_root = tmp_path / "corrected_transcriptions"

    _write(
        asr_root / "2025" / "vid2.srt",
        "1\n00:00:01,000 --> 00:00:02,500\n政府係接納仗委會嘅建議\n",
    )
    _write(
        zh_root / "2025" / "vid2.zh-HK.srt",
        "1\n00:00:01,000 --> 00:00:02,500\n政府接納帳委會建議\n",
    )
    _write(
        yue_root / "2025" / "vid2.yue.srt",
        "1\n00:00:01,000 --> 00:00:02,500\n亂寫字幕\n",
    )

    manifest_out = tmp_path / "logs" / "manifest.csv"
    report_out = tmp_path / "logs" / "report.json"
    yue_report_out = tmp_path / "logs" / "yue_report.csv"

    run_year_correction(
        year="2025",
        asr_root=asr_root,
        zh_root=zh_root,
        yue_root=yue_root,
        output_root=output_root,
        manifest_out=manifest_out,
        report_out=report_out,
        yue_report_out=yue_report_out,
        backend="vllm",
        model_name="dummy",
        ollama_host="http://127.0.0.1:11434",
        overwrite=False,
        max_files=0,
        seed=42,
        review_sample_size=10,
        llm_batch_size=4,
        min_alignment_score=0.2,
        max_alignment_gap_ms=4000,
        vllm_gpu_memory_utilization=0.9,
        vllm_tensor_parallel_size=1,
        max_new_tokens=64,
        temperature=0.0,
        model=DummyCorrectionModel(),
    )

    corrected = (output_root / "2025" / "vid2.srt").read_text(encoding="utf-8")
    assert "政府接納帳委會建議" in corrected

    with manifest_out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["status"] == "corrected"
    assert int(rows[0]["changed_cues"]) == 1
