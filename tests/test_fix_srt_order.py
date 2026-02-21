from __future__ import annotations

from pathlib import Path

from gigacan.subtitle_correction.srt_utils import SubtitleCue

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from importlib import import_module

fix_mod = import_module("7_fix_srt_order")
is_sorted = fix_mod.is_sorted
fix_directory = fix_mod.fix_directory


def _make_srt(path: Path, timestamps: list[tuple[int, int]], text: str = "hello") -> None:
    lines = []
    for idx, (start, end) in enumerate(timestamps, 1):
        h1, r = divmod(start, 3_600_000)
        m1, r = divmod(r, 60_000)
        s1, ms1 = divmod(r, 1_000)
        h2, r = divmod(end, 3_600_000)
        m2, r = divmod(r, 60_000)
        s2, ms2 = divmod(r, 1_000)
        lines.append(f"{idx}")
        lines.append(f"{h1:02}:{m1:02}:{s1:02},{ms1:03} --> {h2:02}:{m2:02}:{s2:02},{ms2:03}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_is_sorted_returns_true_for_ordered_cues() -> None:
    cues = [
        SubtitleCue(1000, 2000, "a"),
        SubtitleCue(3000, 4000, "b"),
        SubtitleCue(5000, 6000, "c"),
    ]
    assert is_sorted(cues)


def test_is_sorted_returns_false_for_unordered_cues() -> None:
    cues = [
        SubtitleCue(5000, 6000, "c"),
        SubtitleCue(1000, 2000, "a"),
        SubtitleCue(3000, 4000, "b"),
    ]
    assert not is_sorted(cues)


def test_fix_directory_sorts_out_of_order_file(tmp_path: Path) -> None:
    srt = tmp_path / "test.srt"
    _make_srt(srt, [(5000, 6000), (1000, 2000), (3000, 4000)])

    fix_directory(tmp_path)

    from gigacan.subtitle_correction.srt_utils import parse_srt

    cues = parse_srt(srt)
    assert [c.start_ms for c in cues] == [1000, 3000, 5000]


def test_fix_directory_skips_already_sorted_file(tmp_path: Path) -> None:
    srt = tmp_path / "sorted.srt"
    _make_srt(srt, [(1000, 2000), (3000, 4000), (5000, 6000)])
    mtime_before = srt.stat().st_mtime_ns

    fix_directory(tmp_path)

    assert srt.stat().st_mtime_ns == mtime_before
