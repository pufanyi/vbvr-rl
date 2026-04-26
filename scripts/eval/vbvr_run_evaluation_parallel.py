#!/usr/bin/env python
"""Compatibility wrapper for ``python -m src.eval.vbvr_run_evaluation_parallel``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.vbvr_run_evaluation_parallel import main

if __name__ == "__main__":
    raise SystemExit(main())
