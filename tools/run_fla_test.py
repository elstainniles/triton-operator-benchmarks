#!/usr/bin/env python3
"""Run one imported FLA test without external tensor fixtures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opbench.fla_test_runner import main
if __name__ == '__main__':
    raise SystemExit(main())
