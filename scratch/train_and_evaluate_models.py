"""
Model Training & Validation Evaluation Pipeline for Submission 2 (Step 5, 6 & 7).

Trains:
1. Improved LightGBM (v2 features + hard negatives)
2. XGBoost (v2 features + hard negatives)

Evaluates threshold sweeps [0.80 to 0.995] & ambiguity margin filtering on 10k validation split.
Reports full telemetry, Macro F0.5, Precision, Recall, TP, FP, FN, Unmatched count.
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
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
from evaluate import compute_entity_metrics, evaluate_predictions
from normalize import normalize_name, normalize_address, normalize_country

VAL_DIR = BASE_DIR / "scratch" / "val_split"
SCRATCH_DIR = BASE_DIR / "scratch"
MODELS_DIR = BASE_DIR / "models"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def train_lgb_v2(X_train, y_train):
    print("\n[TRAINING] LightGBM Model v2...", flush=True)
    t0 = time.time()

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - t0
    joblib.dump(model, MODELS_DIR / "lightgbm_model_v2.joblib")
    print(f"LightGBM v2 trained in {elapsed:.2f}s and saved to models/lightgbm_model_v2.joblib", flush=True)
    return model

def train_xgb_v2(X_train, y_train):
    print("\n[TRAINING] XGBoost Model v2...", flush=True)
    t0 = time.time()

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - t0
    joblib.dump(model, MODELS_DIR / "xgboost_model_v2.joblib")
    print(f"XGBoost v2 trained in {elapsed:.2f}s and saved to models/xgboost_model_v2.joblib", flush=True)
    return model

def run_validation_evaluation(model, model_name: str, blocker, target_store, s1_recs, gt_dict, total_gt_matches, thresholds):
    print(f"\n" + "=" * 70, flush=True)
    print(f"EVALUATING MODEL: {model_name} ON 10K VALIDATION SET", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    val_cache_path = SCRATCH_DIR / "val_features_cache_v2.joblib"
    total_s1 = len(s1_recs)
    prob_cache = {}

    if val_cache_path.exists():
        print("[CACHE HIT] Loading pre-extracted validation features from scratch/val_features_cache_v2.joblib...", flush=True)
        cached_data = joblib.load(val_cache_path)
        s1_cands_list = cached_data['s1_cands_list']

        # Flatten X matrices into a single batch prediction call
        all_X_list = []
        s1_ranges = []
        curr_idx = 0

        for s1_id, cands_list, X in s1_cands_list:
            if cands_list:
                all_X_list.append(X)
                n_rows = len(cands_list)
                s1_ranges.append((s1_id, cands_list, curr_idx, curr_idx + n_rows))
                curr_idx += n_rows
            else:
                s1_ranges.append((s1_id, [], 0, 0))

        if all_X_list:
            X_all = np.vstack(all_X_list)
            print(f"[PREDICT BATCH] Scoring {X_all.shape[0]:,} candidate rows in a single batch call with {model_name}...", flush=True)
            all_probs = model.predict_proba(X_all)[:, 1]

            for s1_id, cands_list, start_i, end_i in s1_ranges:
                if cands_list:
                    probs = all_probs[start_i:end_i]
                    prob_cache[s1_id] = list(zip(cands_list, probs))
                else:
                    prob_cache[s1_id] = []
        else:
            for s1_id, _, _, _ in s1_ranges:
                prob_cache[s1_id] = []
    else:
        print("[EXTRACTING FEATURES] First run: extracting and caching 10k validation features...", flush=True)
        s1_cands_list = []
        candidate_pairs_count = 0
        gt_pairs_found = 0
        last_print = time.time()

        for idx, s1_rec in enumerate(s1_recs):
            s1_id = s1_rec["entity_id"]
            c = s1_rec.get("country", "")
            rn = s1_rec.get("business_name", "")
            ra = s1_rec.get("business_address", "")

            cands = blocker.get_candidates(s1_rec)
            candidate_pairs_count += len(cands)

            gt_set = gt_dict.get(s1_id, set())
            found_gt = len(cands & gt_set)
            gt_pairs_found += found_gt

            s1_prep = precompute_s1_v2(s1_id, c, rn, ra)

            if cands:
                cands_list = list(cands)
                X = np.zeros((len(cands_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
                for r_idx, tid in enumerate(cands_list):
                    fill_features_v2(X, r_idx, s1_prep, target_store, tid)

                probs = model.predict_proba(X)[:, 1]
                prob_cache[s1_id] = list(zip(cands_list, probs))
                s1_cands_list.append((s1_id, cands_list, X))
            else:
                prob_cache[s1_id] = []
                s1_cands_list.append((s1_id, [], None))

            if (idx + 1) % 2500 == 0 or (idx + 1) == total_s1 or time.time() - last_print > 5.0:
                last_print = time.time()
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                pid, ram, cpu = get_telemetry()
                print(f"[{model_name} FEATURE EXTRACTION] S1: {idx+1}/{total_s1} ({((idx+1)/total_s1)*100:.1f}%) | "
                      f"Rate: {rate:.1f} S1/s | Elapsed: {elapsed:.1f}s | RAM: {ram:.2f}GB | PID: {pid}", flush=True)

        print("[SAVING CACHE] Saving extracted features to scratch/val_features_cache_v2.joblib...", flush=True)
        joblib.dump({'s1_cands_list': s1_cands_list}, val_cache_path)

    print("\n" + "-" * 85, flush=True)
    print(f"| {'Model':<15} | {'Thresh':<7} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Matched S1':<10} |", flush=True)
    print("-" * 85, flush=True)

    best_thresh = 0.98
    best_f05 = 0.0
    best_predictions = None

    for thresh in thresholds:
        predictions = {}
        for s1_id, cand_probs in prob_cache.items():
            matches = {tid for tid, p in cand_probs if p >= thresh}
            predictions[s1_id] = matches

        metrics = evaluate_predictions(predictions, gt_dict, beta=0.5)
        f05 = metrics['macro_f05']
        p = metrics['macro_precision']
        r = metrics['macro_recall']
        matched_cnt = sum(1 for m in predictions.values() if len(m) > 0)

        print(f"| {model_name:<15} | {thresh:<7.3f} | {f05:<10.4f} | {p:<9.4f} | {r:<8.4f} | {matched_cnt:<10,} |", flush=True)

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_predictions = predictions

    print("-" * 85, flush=True)
    print(f"BEST THRESHOLD FOR {model_name}: {best_thresh:.3f} (Macro F0.5 = {best_f05:.4f})", flush=True)

    return best_thresh, best_f05, prob_cache

def test_ambiguity_filtering(model_name: str, prob_cache: dict, gt_dict: dict, base_thresh: float):
    print(f"\n[EXPERIMENT] Testing Ambiguity / Margin Filtering on {model_name} @ base threshold {base_thresh:.3f}...", flush=True)
    print("-" * 85, flush=True)
    print(f"| {'Margin Rules':<35} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Matched S1':<10} |", flush=True)
    print("-" * 85, flush=True)

    margins = [0.0, 0.02, 0.05, 0.10, 0.15]

    for m_val in margins:
        predictions = {}
        for s1_id, cand_probs in prob_cache.items():
            # Sort candidate probabilities descending
            sorted_cp = sorted(cand_probs, key=lambda x: x[1], reverse=True)
            high_cands = [cp for cp in sorted_cp if cp[1] >= base_thresh]

            if len(high_cands) >= 2 and m_val > 0.0:
                top1_p = high_cands[0][1]
                top2_p = high_cands[1][1]
                margin = top1_p - top2_p
                if margin < m_val:
                    # Ambiguous top candidates — reject match to protect precision
                    matches = set()
                else:
                    matches = {cp[0] for cp in high_cands}
            else:
                matches = {cp[0] for cp in high_cands}

            predictions[s1_id] = matches

        metrics = evaluate_predictions(predictions, gt_dict, beta=0.5)
        f05 = metrics['macro_f05']
        p = metrics['macro_precision']
        r = metrics['macro_recall']
        matched_cnt = sum(1 for m in predictions.values() if len(m) > 0)

        label = f"Base Thresh {base_thresh:.3f} + Margin > {m_val:.2f}"
        print(f"| {label:<35} | {f05:<10.4f} | {p:<9.4f} | {r:<8.4f} | {matched_cnt:<10,} |", flush=True)

    print("-" * 85, flush=True)

def main():
    print("=" * 70, flush=True)
    print("MODEL TRAINING & THRESHOLD OPTIMIZATION BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Load Training Matrix
    X_train = np.load(SCRATCH_DIR / "X_train_v2.npy")
    y_train = np.load(SCRATCH_DIR / "y_train_v2.npy")
    print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}", flush=True)

    # Step 2: Train LightGBM & XGBoost Models
    lgb_model = train_lgb_v2(X_train, y_train)
    xgb_model = train_xgb_v2(X_train, y_train)

    # Step 3: Load 10k Validation Set
    val_s1_df = pd.read_csv(VAL_DIR / "val_s1.tsv", sep="\t", dtype=str).fillna("")
    val_targets_df = pd.read_csv(VAL_DIR / "val_targets.tsv", sep="\t", dtype=str).fillna("")
    val_gt_df = pd.read_csv(VAL_DIR / "val_gt.tsv", sep="\t", dtype=str).fillna("")

    gt_dict = {}
    for _, row in val_gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        matches = {m.strip() for m in m_str.split(",") if m.strip()} if m_str else set()
        gt_dict[s1_id] = matches

    total_gt_matches = sum(len(s) for s in gt_dict.values())
    s1_recs = val_s1_df.to_dict("records")
    target_recs = val_targets_df.to_dict("records")

    print("\n[PREPARING VALIDATION BLOCKER & TARGET STORE...]", flush=True)
    blocker = AdvancedBlocker(max_posting_size=100, use_prefix3=True)
    target_store = OnDemandTargetStoreV2()

    blocker.build_index(target_recs)
    for rec in target_recs:
        mid = rec["entity_id"]
        c = normalize_country(rec.get("country", ""))
        nn = normalize_name(rec.get("business_name", ""))
        na = normalize_address(rec.get("business_address", ""))
        target_store.add_target_raw(mid, c, nn, na)

    # Step 4: Evaluate Models & Sweep Thresholds
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]

    lgb_thresh, lgb_f05, lgb_probs = run_validation_evaluation(
        lgb_model, "LightGBM_v2", blocker, target_store, s1_recs, gt_dict, total_gt_matches, thresholds
    )

    xgb_thresh, xgb_f05, xgb_probs = run_validation_evaluation(
        xgb_model, "XGBoost_v2", blocker, target_store, s1_recs, gt_dict, total_gt_matches, thresholds
    )

    # Step 5: Test Ambiguity Filtering
    test_ambiguity_filtering("LightGBM_v2", lgb_probs, gt_dict, lgb_thresh)
    test_ambiguity_filtering("XGBoost_v2", xgb_probs, gt_dict, xgb_thresh)

    print("\n" + "=" * 70, flush=True)
    print("MODEL & THRESHOLD OPTIMIZATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"LightGBM v2 Best Threshold : {lgb_thresh:.3f} | Macro F0.5 = {lgb_f05:.4f}", flush=True)
    print(f"XGBoost v2 Best Threshold  : {xgb_thresh:.3f} | Macro F0.5 = {xgb_f05:.4f}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
