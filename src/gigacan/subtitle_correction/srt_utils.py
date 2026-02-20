from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from gigacan.qwen_srt.srt import write_srt


TIMESTAMP_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})$"
)


@dataclass(slots=True, frozen=True)
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


def _parse_timestamp_to_ms(value: str) -> int:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt_content(content: str) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    if not content.strip():
        return cues

    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if lines[0].strip().isdigit():
            lines = lines[1:]
        if not lines:
            continue

        timestamp_line = lines[0].strip().lstrip("\ufeff")
        match = TIMESTAMP_RE.match(timestamp_line)
        if match is None:
            continue

        start_ms = _parse_timestamp_to_ms(match.group(1))
        end_ms = _parse_timestamp_to_ms(match.group(2))
        if end_ms <= start_ms:
            continue

        text_lines = lines[1:]
        text = "\n".join(text_lines).strip()
        if not text:
            continue

        cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))

    return cues


def parse_srt(path: Path) -> list[SubtitleCue]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return parse_srt_content(content)


def write_srt_cues(path: Path, cues: list[SubtitleCue]) -> None:
    entries = [(cue.start_ms, cue.end_ms, cue.text) for cue in cues]
    write_srt(path, entries)
