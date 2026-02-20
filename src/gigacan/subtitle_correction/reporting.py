from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .normalize import TextNormalizer, tokenize_terms
from .srt_utils import SubtitleCue


@dataclass(slots=True, frozen=True)
class YueDriftRow:
    video_id: str
    year: str
    has_zh_hk_ref: bool
    cue_count_asr: int
    cue_count_yue: int
    coverage_time_overlap_ratio: float
    normalized_text_similarity: float
    char_length_ratio_yue_to_asr: float
    top_diff_terms: str
    notes: str


def _interval_overlap_ms(a: SubtitleCue, b: SubtitleCue) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def compute_time_overlap_ratio(asr_cues: list[SubtitleCue], yue_cues: list[SubtitleCue]) -> float:
    total_asr_ms = sum(max(0, cue.end_ms - cue.start_ms) for cue in asr_cues)
    if total_asr_ms <= 0:
        return 0.0

    covered_ms = 0
    yue_index = 0
    for asr_cue in asr_cues:
        while yue_index < len(yue_cues) and yue_cues[yue_index].end_ms <= asr_cue.start_ms:
            yue_index += 1

        probe = yue_index
        best = 0
        while probe < len(yue_cues):
            yue_cue = yue_cues[probe]
            if yue_cue.start_ms >= asr_cue.end_ms:
                break
            best = max(best, _interval_overlap_ms(asr_cue, yue_cue))
            probe += 1
        covered_ms += best

    return covered_ms / total_asr_ms


def _collect_term_counter(cues: list[SubtitleCue], normalizer: TextNormalizer) -> Counter[str]:
    counter: Counter[str] = Counter()
    for cue in cues:
        normalized = normalizer.clean(cue.text)
        for term in tokenize_terms(normalized):
            if len(term) <= 1:
                continue
            counter[term] += 1
    return counter


def _top_diff_terms(
    asr_cues: list[SubtitleCue],
    yue_cues: list[SubtitleCue],
    normalizer: TextNormalizer,
    *,
    limit: int = 8,
) -> str:
    asr_counts = _collect_term_counter(asr_cues, normalizer)
    yue_counts = _collect_term_counter(yue_cues, normalizer)

    all_terms = set(asr_counts) | set(yue_counts)
    if not all_terms:
        return ""

    diffs = []
    for term in all_terms:
        delta = abs(asr_counts.get(term, 0) - yue_counts.get(term, 0))
        if delta <= 0:
            continue
        diffs.append((delta, term))

    if not diffs:
        return ""

    diffs.sort(key=lambda item: (-item[0], item[1]))
    return " | ".join(term for _, term in diffs[:limit])


def build_yue_drift_row(
    *,
    video_id: str,
    year: str,
    asr_cues: list[SubtitleCue],
    yue_cues: list[SubtitleCue],
    has_zh_hk_ref: bool,
    normalizer: TextNormalizer,
) -> YueDriftRow:
    asr_text = " ".join(normalizer.for_similarity(cue.text) for cue in asr_cues)
    yue_text = " ".join(normalizer.for_similarity(cue.text) for cue in yue_cues)

    if asr_text and yue_text:
        similarity = SequenceMatcher(a=asr_text, b=yue_text).ratio()
    else:
        similarity = 0.0

    if asr_text:
        length_ratio = len(yue_text) / len(asr_text)
    else:
        length_ratio = 0.0

    overlap = compute_time_overlap_ratio(asr_cues, yue_cues)

    notes: list[str] = []
    if overlap < 0.2:
        notes.append("low_time_overlap")
    if similarity < 0.25:
        notes.append("low_text_similarity")
    if len(yue_cues) < max(1, len(asr_cues) // 4):
        notes.append("likely_condensed_caption")

    return YueDriftRow(
        video_id=video_id,
        year=year,
        has_zh_hk_ref=has_zh_hk_ref,
        cue_count_asr=len(asr_cues),
        cue_count_yue=len(yue_cues),
        coverage_time_overlap_ratio=round(overlap, 6),
        normalized_text_similarity=round(similarity, 6),
        char_length_ratio_yue_to_asr=round(length_ratio, 6),
        top_diff_terms=_top_diff_terms(asr_cues, yue_cues, normalizer),
        notes="|".join(notes),
    )
