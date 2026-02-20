#!/usr/bin/env python3
"""CLI entrypoint for zh-HK-guided subtitle correction and yue drift reporting."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gigacan.subtitle_correction.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
