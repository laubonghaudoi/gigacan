from __future__ import annotations

import numpy as np
import pytest

from gigacan.qwen_srt.audio import slice_audio_segment


def test_slice_audio_segment_returns_expected_window() -> None:
    audio = np.arange(0, 16000, dtype=np.float32)
    segment = slice_audio_segment(audio, 16000, 250, 500)
    assert segment.shape == (4000,)
    assert float(segment[0]) == 4000.0
    assert float(segment[-1]) == 7999.0


def test_slice_audio_segment_clamps_end_to_audio_length() -> None:
    audio = np.arange(0, 1000, dtype=np.float32)
    segment = slice_audio_segment(audio, 1000, 900, 2000)
    assert segment.shape == (100,)
    assert float(segment[0]) == 900.0
    assert float(segment[-1]) == 999.0


def test_slice_audio_segment_rejects_invalid_range() -> None:
    audio = np.arange(0, 100, dtype=np.float32)
    with pytest.raises(ValueError):
        slice_audio_segment(audio, 16000, 200, 100)
