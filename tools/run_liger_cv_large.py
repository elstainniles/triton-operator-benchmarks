"""Portable large-case entry point; plan/verify need only Python 3.11+."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opbench.liger_cv_large import main

if __name__ == '__main__':
    main()
