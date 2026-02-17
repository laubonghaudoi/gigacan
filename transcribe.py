#!/usr/bin/env python3
"""CLI entrypoint for local Qwen3-ASR transcription to SRT."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gigacan.qwen_srt.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
