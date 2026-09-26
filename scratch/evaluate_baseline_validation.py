"""
Baseline Validation Evaluator for Submission 2 Development Loop.

Evaluates the exact Submission 1 baseline (Strategy 3 MultiKey Blocker + 13 Features + Baseline LightGBM @ threshold 0.98)
against the 10,000 S1 leakage-free validation split in scratch/val_split/.
Continuous numerical telemetry is logged every batch.
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution" / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from blocking import EntityBlocker
from inference_optimized import OnDemandTargetStore, precompute_s1_record, fill_features_optimized, fast_norm_name, fast_norm_addr
from evaluate import compute_entity_metrics, evaluate_predictions

VAL_DIR = BASE_DIR / "scratch" / "val_split"
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.joblib"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def main():
    print("=" * 70)
    print("EVALUATING BASELINE (SUBMISSION 1 PIPELINE) ON 10K VALIDATION SET")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Load Ground Truth and Validation Data
    val_s1_df = pd.read_csv(VAL_DIR / "val_s1.tsv", sep="\t", dtype=str).fillna("")
    val_targets_df = pd.read_csv(VAL_DIR / "val_targets.tsv", sep="\t", dtype=str).fillna("")
    val_gt_df = pd.read_csv(VAL_DIR / "val_gt.tsv", sep="\t", dtype=str).fillna("")

    gt_dict = {}
    for _, row in val_gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        matches = set(m_str.split(",")) if m_str else set()
        gt_dict[s1_id] = {m for m in matches if m}

    total_gt_matches = sum(len(s) for s in gt_dict.values())
    print(f"Validation S1: {len(val_s1_df):,}", flush=True)
    print(f"Validation Target Pool: {len(val_targets_df):,}", flush=True)
    print(f"Total True GT Match Pairs: {total_gt_matches:,}", flush=True)

    # Step 2: Build Blocker Index & OnDemand Store
    print("\n[STAGE 1/3] Building Blocker Index & Target Store...")
    blocker = EntityBlocker()
    target_store = OnDemandTargetStore()

    target_recs = val_targets_df.to_dict("records")
    blocker.build_index(target_recs)

    for rec in target_recs:
        mid = rec["entity_id"]
        c = rec.get("country", "")
        nn = fast_norm_name(rec.get("business_name", ""))
        na = fast_norm_addr(rec.get("business_address", ""))
        target_store.add_target_raw(mid, c, nn, na)

    # Step 3: Load LightGBM Model
    print("\n[STAGE 2/3] Loading LightGBM baseline model...")
    lgb_model = joblib.load(MODEL_PATH)

    # Step 4: Candidate Generation & Feature Extraction
    print("\n[STAGE 3/3] Generating candidates & predicting matches...")
    s1_recs = val_s1_df.to_dict("records")

    total_s1 = len(s1_recs)
    candidate_pairs_count = 0
    gt_pairs_found = 0

    predictions = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    batch_size = 1000
    last_print = time.time()

    for idx, s1_rec in enumerate(s1_recs):
        s1_id = s1_rec["entity_id"]
        c = s1_rec.get("country", "")
        rn = s1_rec.get("business_name", "")
        ra = s1_rec.get("business_address", "")

        cands = blocker.get_candidates(s1_rec)
        candidate_pairs_count += len(cands)

        # Check candidate recall
        gt_set = gt_dict.get(s1_id, set())
        found_gt = len(cands & gt_set)
        gt_pairs_found += found_gt

        s1_prep = precompute_s1_record(s1_id, c, rn, ra)

        match_preds = set()
        if cands:
            cands_list = list(cands)
            X = np.zeros((len(cands_list), 13), dtype=np.float32)
            for r_idx, tid in enumerate(cands_list):
                fill_features_optimized(X, r_idx, s1_prep, target_store, tid)

            probs = lgb_model.predict_proba(X)[:, 1]
            for r_idx, tid in enumerate(cands_list):
                if probs[r_idx] >= 0.98:
                    match_preds.add(tid)

        predictions[s1_id] = match_preds

        # Telemetry every 2,000 S1 or every 5s
        if (idx + 1) % 2000 == 0 or (idx + 1) == total_s1 or time.time() - last_print > 5.0:
            last_print = time.time()
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_s1 - (idx + 1)) / rate if rate > 0 else 0
            pid, ram, cpu = get_telemetry()
            cand_recall = (gt_pairs_found / total_gt_matches) * 100 if total_gt_matches > 0 else 0
            cands_per_s1 = candidate_pairs_count / (idx + 1)

            print(f"[EVALUATING BASELINE] S1: {idx+1}/{total_s1} ({((idx+1)/total_s1)*100:.1f}%) | "
                  f"Cands: {candidate_pairs_count:,} ({cands_per_s1:.1f}/S1) | "
                  f"Cand Recall: {cand_recall:.2f}% | "
                  f"Rate: {rate:.1f} S1/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | RAM: {ram:.2f}GB | CPU: {cpu}% | PID: {pid}", flush=True)

    # Compute Macro Metrics
    metrics = evaluate_predictions(predictions, gt_dict, beta=0.5)

    # Detailed counts
    pred_matches_cnt = sum(1 for m in predictions.values() if len(m) > 0)
    unmatched_cnt = sum(1 for m in predictions.values() if len(m) == 0)

    print("\n" + "=" * 70)
    print("BASELINE VALIDATION RESULTS (SUBMISSION 1 PIPELINE)")
    print("=" * 70)
    print(f"Macro F0.5 Score   : {metrics['macro_f05']:.4f}")
    print(f"Macro Precision    : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"Candidate Recall   : {(gt_pairs_found / total_gt_matches)*100:.2f}% ({gt_pairs_found}/{total_gt_matches})")
    print(f"Candidate Count    : {candidate_pairs_count:,} ({candidate_pairs_count/total_s1:.1f} cands/S1)")
    print(f"Matched S1 Count   : {pred_matches_cnt:,} (Unmatched: {unmatched_cnt:,})")
    print(f"Total Runtime      : {time.time() - start_time:.2f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()
