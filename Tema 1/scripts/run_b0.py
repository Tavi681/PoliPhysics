#!/usr/bin/env python3
"""Thin wrapper: python scripts/run_b0.py  →  stage_b.py --b0."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "stage_b.py"
raise SystemExit(
    __import__("subprocess").call(
        [sys.executable, "-u", str(SCRIPT), "--b0", *sys.argv[1:]]
    )
)
