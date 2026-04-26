#!/usr/bin/env python
"""Compatibility wrapper for ``python -m src.eval.vbvr_run_evaluation_parallel``."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

main = import_module("src.eval.vbvr_run_evaluation_parallel").main

if __name__ == "__main__":
    raise SystemExit(main())
