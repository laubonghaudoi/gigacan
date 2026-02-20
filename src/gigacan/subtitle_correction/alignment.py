from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .normalize import TextNormalizer
from .srt_utils import SubtitleCue


@dataclass(slots=True, frozen=True)
class CueAlignment:
    asr_index: int
    reference_index: int | None
    score: float
    time_score: float
    text_score: float
    evidence_text: str


def interval_overlap_ms(a: SubtitleCue, b: SubtitleCue) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def interval_gap_ms(a: SubtitleCue, b: SubtitleCue) -> int:
    if a.end_ms < b.start_ms:
        return b.start_ms - a.end_ms
    if b.end_ms < a.start_ms:
        return a.start_ms - b.end_ms
    return 0


def compute_text_similarity(
    asr_text: str,
    reference_text: str,
    normalizer: TextNormalizer,
) -> float:
    left = normalizer.for_similarity(asr_text)
    right = normalizer.for_similarity(reference_text)
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def align_cues(
    asr_cues: list[SubtitleCue],
    reference_cues: list[SubtitleCue],
    *,
    normalizer: TextNormalizer,
    max_gap_ms: int = 4000,
    min_score: float = 0.22,
) -> list[CueAlignment]:
    alignments: list[CueAlignment] = []
    if not asr_cues:
        return alignments
    if not reference_cues:
        return [
            CueAlignment(
                asr_index=index,
                reference_index=None,
                score=0.0,
                time_score=0.0,
                text_score=0.0,
                evidence_text="",
            )
            for index in range(len(asr_cues))
        ]

    ref_start_index = 0
    for asr_index, asr_cue in enumerate(asr_cues):
        while (
            ref_start_index < len(reference_cues)
            and reference_cues[ref_start_index].end_ms < asr_cue.start_ms - max_gap_ms
        ):
            ref_start_index += 1

        best_ref_index: int | None = None
        best_score = 0.0
        best_time_score = 0.0
        best_text_score = 0.0

        ref_index = ref_start_index
        while ref_index < len(reference_cues):
            ref_cue = reference_cues[ref_index]
            if ref_cue.start_ms > asr_cue.end_ms + max_gap_ms:
                break

            overlap_ms = interval_overlap_ms(asr_cue, ref_cue)
            if overlap_ms > 0:
                denom = max(1, max(asr_cue.end_ms - asr_cue.start_ms, ref_cue.end_ms - ref_cue.start_ms))
                time_score = overlap_ms / denom
            else:
                gap_ms = interval_gap_ms(asr_cue, ref_cue)
                if gap_ms > max_gap_ms:
                    ref_index += 1
                    continue
                time_score = max(0.0, 1.0 - (gap_ms / max_gap_ms)) * 0.35

            text_score = compute_text_similarity(asr_cue.text, ref_cue.text, normalizer)
            score = 0.7 * time_score + 0.3 * text_score

            if score > best_score:
                best_ref_index = ref_index
                best_score = score
                best_time_score = time_score
                best_text_score = text_score

            ref_index += 1

        if best_ref_index is None or best_score < min_score:
            alignments.append(
                CueAlignment(
                    asr_index=asr_index,
                    reference_index=None,
                    score=best_score,
                    time_score=best_time_score,
                    text_score=best_text_score,
                    evidence_text="",
                )
            )
            continue

        alignments.append(
            CueAlignment(
                asr_index=asr_index,
                reference_index=best_ref_index,
                score=best_score,
                time_score=best_time_score,
                text_score=best_text_score,
                evidence_text=reference_cues[best_ref_index].text,
            )
        )

    return alignments
