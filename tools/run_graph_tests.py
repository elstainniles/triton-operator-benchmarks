"""Convenience alias for the self-contained graph benchmarks (no .pt required)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'run')
    main()
