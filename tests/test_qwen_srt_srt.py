from __future__ import annotations

from pathlib import Path

from gigacan.qwen_srt.srt import ms_to_srt_time, write_srt


def test_ms_to_srt_time_bounds() -> None:
    assert ms_to_srt_time(-5) == "00:00:00,000"
    assert ms_to_srt_time(3_723_004) == "01:02:03,004"


def test_write_srt_outputs_expected_format(tmp_path: Path) -> None:
    target = tmp_path / "out.srt"
    write_srt(
        target,
        [
            (1000, 2100, "first line"),
            (2200, 5000, "second line"),
        ],
    )
    assert target.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,100\nfirst line\n\n"
        "2\n00:00:02,200 --> 00:00:05,000\nsecond line\n"
    )
