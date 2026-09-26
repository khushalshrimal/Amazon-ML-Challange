"""
Leakage-Free Validation Split Generator for Amazon ML Challenge Submission 2.

Creates a representative, leakage-free 10,000 S1 validation benchmark:
- 10,000 S1 entities sampled randomly with fixed seed (42).
- All true target matches (S2 and S3) from train_ground_truth.tsv.
- ~400,000 distractor target entities (S2 and S3) sampled from train_source2.tsv / train_source3.tsv.
- Saved in scratch/val_split/ for reproducible evaluation across all experiments.
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "scratch" / "val_split"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def main():
    print("=" * 60)
    print("BUILDING LEAKAGE-FREE VALIDATION SPLIT (10,000 S1 Entities)")
    print("=" * 60)
    start_time = time.time()
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Ground Truth
    print("[STAGE 1/4] Loading train ground truth...")
    gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    print(f"Total train S1 in GT: {len(gt_df):,}")

    # Step 2: Sample 10,000 S1 IDs reproducibly
    np.random.seed(42)
    sample_indices = np.random.choice(len(gt_df), size=10000, replace=False)
    val_gt_df = gt_df.iloc[sample_indices].copy().reset_index(drop=True)
    val_s1_set = set(val_gt_df["source1_entity_id"])

    # Extract all true target IDs for validation set
    all_true_targets = set()
    for _, row in val_gt_df.iterrows():
        m_str = row["matched_entity_ids"]
        if m_str:
            for tid in m_str.split(","):
                tid = tid.strip()
                if tid:
                    all_true_targets.add(tid)

    print(f"Sampled {len(val_s1_set):,} validation S1 records.")
    print(f"Total true target entities required in validation target pool: {len(all_true_targets):,}")

    # Step 3: Load S1 records
    print("[STAGE 2/4] Filtering validation S1 records from train_source1.tsv...")
    s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].copy().reset_index(drop=True)
    val_s1_df.to_csv(VAL_DIR / "val_s1.tsv", sep="\t", index=False)
    val_gt_df.to_csv(VAL_DIR / "val_gt.tsv", sep="\t", index=False)
    print(f"Saved val_s1.tsv ({len(val_s1_df):,} rows) and val_gt.tsv ({len(val_gt_df):,} rows).")

    # Step 4: Build Target Pool (True Targets + ~400k Distractors)
    print("[STAGE 3/4] Loading S2 and S3 target pools...")
    s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str)
    s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str)

    # Separate true targets vs available distractors
    s2_true = s2_df[s2_df["entity_id"].isin(all_true_targets)]
    s3_true = s3_df[s3_df["entity_id"].isin(all_true_targets)]

    s2_other = s2_df[~s2_df["entity_id"].isin(all_true_targets)]
    s3_other = s3_df[~s3_df["entity_id"].isin(all_true_targets)]

    # Sample distractors reproducibly
    s2_distractors = s2_other.sample(n=min(200000, len(s2_other)), random_state=42)
    s3_distractors = s3_other.sample(n=min(200000, len(s3_other)), random_state=42)

    val_targets_df = pd.concat([s2_true, s3_true, s2_distractors, s3_distractors], ignore_index=True)
    val_targets_df = val_targets_df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    val_targets_df.to_csv(VAL_DIR / "val_targets.tsv", sep="\t", index=False)

    elapsed = time.time() - start_time
    pid, ram, cpu = get_telemetry()
    print("=" * 60)
    print(f"[SUMMARY]")
    print(f"Validation S1 count: {len(val_s1_df):,}")
    print(f"Validation Target Pool count: {len(val_targets_df):,} (True: {len(all_true_targets):,}, Distractors: {len(val_targets_df) - len(all_true_targets):,})")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"RAM: {ram:.2f} GB | PID: {pid}")
    print("=" * 60)

if __name__ == "__main__":
    main()
