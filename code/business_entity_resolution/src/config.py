"""
Configuration parameters and constants for Amazon ML Challenge 2026 Entity Resolution.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# Dataset Files
TRAIN_S1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1_PATH = TEST_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Open-Set Country Settings
TRAIN_COUNTRIES = ["US", "India"]
TEST_COUNTRIES = ["US", "India", "France"]

# Evaluation Settings
F_BETA = 0.5

# TODO: Add blocking and feature extraction hyperparameter configs in Phase 4-7
