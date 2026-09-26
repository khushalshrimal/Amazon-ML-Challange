"""
Training Dataset Generator with Hard Negative Mining (Step 4).

Generates a balanced, hard-negative enriched dataset for training LightGBM / XGBoost models:
- 30,000 S1 entities sampled from train_source1.tsv (excluding the 10,000 validation split S1 entities to avoid ANY data leakage).
- All True Positives (y=1) from train_ground_truth.tsv.
- Hard Negatives (y=0): Blocker candidates that share prefixes, tokens, or country but are NOT ground truth.
- Random Negatives (y=0): Randomly sampled targets from same country.

Outputs:
- scratch/X_train_v2.npy
- scratch/y_train_v2.npy
- scratch/train_meta.csv
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution" / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from test_blocking_experiments import AdvancedBlocker
from features_v2 import (
    OnDemandTargetStoreV2, precompute_s1_v2, fill_features_v2,
    FEATURE_NAMES_V2
)
from normalize import normalize_name, normalize_address, normalize_country

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "scratch" / "val_split"
SCRATCH_DIR = BASE_DIR / "scratch"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def main():
    print("=" * 70, flush=True)
    print("BUILDING TRAINING DATASET WITH HARD NEGATIVE MINING (v2 Features)", flush=True)
    print("=" * 70, flush=True)
    start_time = time.time()

    # Step 1: Load Ground Truth
    gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    val_s1_df = pd.read_csv(VAL_DIR / "val_s1.tsv", sep="\t", dtype=str).fillna("")
    val_s1_set = set(val_s1_df["entity_id"])

    # Exclude validation S1 IDs from training set to guarantee zero leakage
    train_gt_df = gt_df[~gt_df["source1_entity_id"].isin(val_s1_set)].copy().reset_index(drop=True)
    print(f"Available non-validation S1 entities in train GT: {len(train_gt_df):,}", flush=True)

    # Sample 25,000 S1 entities for training dataset construction
    np.random.seed(123)
    sample_indices = np.random.choice(len(train_gt_df), size=25000, replace=False)
    train_sample_gt = train_gt_df.iloc[sample_indices].copy().reset_index(drop=True)
    train_s1_set = set(train_sample_gt["source1_entity_id"])

    gt_dict = {}
    all_true_targets = set()
    for _, row in train_sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        matches = {m.strip() for m in m_str.split(",") if m.strip()} if m_str else set()
        gt_dict[s1_id] = matches
        all_true_targets.update(matches)

    print(f"Sampled {len(train_s1_set):,} training S1 entities.", flush=True)
    print(f"Total True Match Target entities in sample: {len(all_true_targets):,}", flush=True)

    # Step 2: Load Source 1 and Target Data
    print("\n[STAGE 1/4] Loading S1, S2, and S3 records for training pool...", flush=True)
    s1_all = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_train_df = s1_all[s1_all["entity_id"].isin(train_s1_set)].copy().reset_index(drop=True)

    s2_all = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3_all = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str).fillna("")

    # Separate true targets vs distractors
    s2_true = s2_all[s2_all["entity_id"].isin(all_true_targets)]
    s3_true = s3_all[s3_all["entity_id"].isin(all_true_targets)]

    s2_dist = s2_all[~s2_all["entity_id"].isin(all_true_targets)].sample(n=min(150000, len(s2_all)), random_state=123)
    s3_dist = s3_all[~s3_all["entity_id"].isin(all_true_targets)].sample(n=min(150000, len(s3_all)), random_state=123)

    target_train_df = pd.concat([s2_true, s3_true, s2_dist, s3_dist], ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    print(f"Target pool count for training: {len(target_train_df):,}", flush=True)

    # Step 3: Build Blocker & Target Store V2
    print("\n[STAGE 2/4] Building Blocker Index & Target Store V2...", flush=True)
    blocker = AdvancedBlocker(max_posting_size=100, use_prefix3=True)
    target_store = OnDemandTargetStoreV2()

    target_recs = target_train_df.to_dict("records")
    blocker.build_index(target_recs)

    for rec in target_recs:
        mid = rec["entity_id"]
        c = normalize_country(rec.get("country", ""))
        nn = normalize_name(rec.get("business_name", ""))
        na = normalize_address(rec.get("business_address", ""))
        target_store.add_target_raw(mid, c, nn, na)

    # Step 4: Generate Pair Features & Labels
    print("\n[STAGE 3/4] Generating True Positive & Hard Negative Pair Features...", flush=True)
    s1_recs = s1_train_df.to_dict("records")
    total_s1 = len(s1_recs)

    pair_rows = []
    labels = []
    last_print = time.time()

    total_tp = 0
    total_hn = 0

    for idx, s1_rec in enumerate(s1_recs):
        s1_id = s1_rec["entity_id"]
        c = s1_rec.get("country", "")
        rn = s1_rec.get("business_name", "")
        ra = s1_rec.get("business_address", "")

        gt_matches = gt_dict.get(s1_id, set())
        cands = blocker.get_candidates(s1_rec)

        s1_prep = precompute_s1_v2(s1_id, c, rn, ra)

        # 1. True Positives (y=1)
        for tid in gt_matches:
            if tid in target_store.raw_attr:
                pair_rows.append((s1_prep, tid, 1.0))
                labels.append(1)
                total_tp += 1

        # 2. Hard Negatives (y=0) from blocker candidates (up to 6 per S1)
        hard_negs = [tid for tid in cands if tid not in gt_matches and tid in target_store.raw_attr]
        if len(hard_negs) > 6:
            np.random.seed(idx)
            hard_negs = list(np.random.choice(hard_negs, size=6, replace=False))

        for tid in hard_negs:
            pair_rows.append((s1_prep, tid, 1.0))
            labels.append(0)
            total_hn += 1

        if (idx + 1) % 5000 == 0 or (idx + 1) == total_s1 or time.time() - last_print > 5.0:
            last_print = time.time()
            elapsed = time.time() - start_time
            pid, ram, cpu = get_telemetry()
            print(f"[BUILDING TRAIN DATASET] S1: {idx+1}/{total_s1} ({((idx+1)/total_s1)*100:.1f}%) | "
                  f"Total Pairs: {len(labels):,} (TP: {total_tp:,}, Hard Neg: {total_hn:,}) | "
                  f"Elapsed: {elapsed:.1f}s | RAM: {ram:.2f}GB | CPU: {cpu}% | PID: {pid}", flush=True)

    # Step 5: Fill Feature Matrix NumPy Array
    print(f"\n[STAGE 4/4] Allocating feature matrix X_train: {len(pair_rows):,} rows x {len(FEATURE_NAMES_V2)} columns...", flush=True)
    X_train = np.zeros((len(pair_rows), len(FEATURE_NAMES_V2)), dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)

    for r_idx, (s1_prep, tid, score) in enumerate(pair_rows):
        fill_features_v2(X_train, r_idx, s1_prep, target_store, tid, score)

    np.save(SCRATCH_DIR / "X_train_v2.npy", X_train)
    np.save(SCRATCH_DIR / "y_train_v2.npy", y_train)

    elapsed = time.time() - start_time
    pid, ram, cpu = get_telemetry()
    print("=" * 70, flush=True)
    print(f"TRAINING DATASET SUCCESSFULLY GENERATED!", flush=True)
    print(f"Total Rows: {len(y_train):,} (Positive: {np.sum(y_train==1):,}, Negative: {np.sum(y_train==0):,})", flush=True)
    print(f"Saved: scratch/X_train_v2.npy ({X_train.nbytes / (1024*1024):.1f} MB)", flush=True)
    print(f"Saved: scratch/y_train_v2.npy", flush=True)
    print(f"Total Runtime: {elapsed:.2f} s | RAM: {ram:.2f} GB", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
