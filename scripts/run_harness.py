#!/usr/bin/env python3
"""Convenience launcher for the installed package: ``python scripts/run_harness.py ...``"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_harness.loop import main  # noqa: E402

if __name__ == "__main__":
    main()
