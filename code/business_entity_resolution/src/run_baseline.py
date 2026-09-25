"""
Script to execute baseline validation metrics and blocking benchmarks.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from evaluate import evaluate_predictions, run_evaluation_unit_tests

def main():
    print("=== EXECUTING PHASE 3 BASELINE EVALUATIONS ===")
    run_evaluation_unit_tests()
    print("Phase 3 baseline validation runner finished successfully.")

if __name__ == "__main__":
    main()
