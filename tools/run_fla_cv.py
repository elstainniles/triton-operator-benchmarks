"""FLA CV workbook cases: plan, verify, run, and resume."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opbench.fla_cv import main

if __name__ == '__main__':
    main()
