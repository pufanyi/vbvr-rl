#!/usr/bin/env python
"""Compatibility wrapper for ``python -m src.eval.build_vbvr_eval_json``."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

main = import_module("src.eval.build_vbvr_eval_json").main

if __name__ == "__main__":
    raise SystemExit(main())
