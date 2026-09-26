"""
Ensemble Model & Fine Threshold Sweep Benchmark for Submission 2.

Evaluates an ensemble blend: P_ensemble = 0.5 * P_lgb + 0.5 * P_xgb
Sweeps fine threshold values: [0.970, 0.975, 0.980, 0.985, 0.988, 0.990, 0.992, 0.994, 0.995, 0.996, 0.997]
Logs complete evaluation results and identifies the winning frozen pipeline.
"""

import os
import sys
import time
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

from evaluate import compute_entity_metrics, evaluate_predictions

VAL_DIR = BASE_DIR / "scratch" / "val_split"
SCRATCH_DIR = BASE_DIR / "scratch"
MODELS_DIR = BASE_DIR / "models"

def main():
    print("=" * 70, flush=True)
    print("ENSEMBLE MODEL & FINE THRESHOLD BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Load Models & Feature Cache
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_model_v2.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_v2.joblib")
    cached_data = joblib.load(SCRATCH_DIR / "val_features_cache_v2.joblib")
    s1_cands_list = cached_data['s1_cands_list']

    val_gt_df = pd.read_csv(VAL_DIR / "val_gt.tsv", sep="\t", dtype=str).fillna("")
    gt_dict = {}
    for _, row in val_gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        matches = {m.strip() for m in m_str.split(",") if m.strip()} if m_str else set()
        gt_dict[s1_id] = matches

    # Step 2: Batch Predict Probabilities
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

    X_all = np.vstack(all_X_list)
    print(f"Scoring {X_all.shape[0]:,} candidate rows with LightGBM and XGBoost...", flush=True)

    probs_lgb = lgb_model.predict_proba(X_all)[:, 1]
    probs_xgb = xgb_model.predict_proba(X_all)[:, 1]
    probs_ens = 0.5 * probs_lgb + 0.5 * probs_xgb

    prob_cache_lgb = {}
    prob_cache_xgb = {}
    prob_cache_ens = {}

    for s1_id, cands_list, start_i, end_i in s1_ranges:
        if cands_list:
            prob_cache_lgb[s1_id] = list(zip(cands_list, probs_lgb[start_i:end_i]))
            prob_cache_xgb[s1_id] = list(zip(cands_list, probs_xgb[start_i:end_i]))
            prob_cache_ens[s1_id] = list(zip(cands_list, probs_ens[start_i:end_i]))
        else:
            prob_cache_lgb[s1_id] = []
            prob_cache_xgb[s1_id] = []
            prob_cache_ens[s1_id] = []

    # Step 3: Sweep Fine Thresholds
    fine_thresholds = [0.970, 0.975, 0.980, 0.985, 0.988, 0.990, 0.992, 0.994, 0.995, 0.996, 0.997]

    for model_name, prob_cache in [("LightGBM_v2", prob_cache_lgb), ("XGBoost_v2", prob_cache_xgb), ("Ensemble_v2", prob_cache_ens)]:
        print("\n" + "-" * 85, flush=True)
        print(f"FINE THRESHOLD SWEEP FOR: {model_name}", flush=True)
        print("-" * 85, flush=True)
        print(f"| {'Model':<15} | {'Thresh':<7} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Matched S1':<10} |", flush=True)
        print("-" * 85, flush=True)

        best_t = 0.98
        best_f05 = 0.0
        best_p = 0.0
        best_r = 0.0

        for t in fine_thresholds:
            preds = {}
            for s1_id, c_probs in prob_cache.items():
                preds[s1_id] = {tid for tid, p in c_probs if p >= t}

            metrics = evaluate_predictions(preds, gt_dict, beta=0.5)
            f05 = metrics['macro_f05']
            p = metrics['macro_precision']
            r = metrics['macro_recall']
            matched_cnt = sum(1 for m in preds.values() if len(m) > 0)

            print(f"| {model_name:<15} | {t:<7.3f} | {f05:<10.4f} | {p:<9.4f} | {r:<8.4f} | {matched_cnt:<10,} |", flush=True)

            if f05 > best_f05:
                best_f05 = f05
                best_t = t
                best_p = p
                best_r = r

        print("-" * 85, flush=True)
        print(f">>> BEST FOR {model_name}: Threshold={best_t:.3f} | Macro F0.5={best_f05:.4f} | Precision={best_p:.4f} | Recall={best_r:.4f}", flush=True)

if __name__ == "__main__":
    main()
