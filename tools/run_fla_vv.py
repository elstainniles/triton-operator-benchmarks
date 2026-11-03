"""Ordered FLA VV batches; run from the repository root."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opbench.fla_vv import main

if __name__ == '__main__':
    raise SystemExit(main())
