"""
Submission 2 Full Test Inference Script (Step 17).

Runs full test set inference across:
- S1 records: 1,732,544 (test_source1.tsv)
- Target pool: 9,969,589 (test_source2.tsv + test_source3.tsv)

Pipeline:
1. AdvancedBlocker (max_posting_size=100, use_prefix3=True, stop-words)
2. OnDemandTargetStoreV2 (32 features)
3. Ensemble Model: 0.5 * LightGBM_v2 + 0.5 * XGBoost_v2
4. Threshold: 0.995

Generates:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Provides mandatory continuous live telemetry every batch.
"""

import os
import sys
import time
import gc
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
from normalize import normalize_name, normalize_address, normalize_country

TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def main():
    print("=" * 80, flush=True)
    print("SUBMISSION 2 FULL TEST INFERENCE (1.73M S1 Entities)", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Trained Models
    print("[STAGE 1/4] Loading LightGBM v2 and XGBoost v2 trained models...", flush=True)
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_model_v2.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_v2.joblib")
    print("Models loaded successfully.", flush=True)

    # Step 2: Load Target Pool & Build Blocker Index
    print("\n[STAGE 2/4] Loading Test Target Pool (S2 + S3 = ~9.97M records)...", flush=True)
    t0_block = time.time()

    blocker = AdvancedBlocker(max_posting_size=100, use_prefix3=True)
    target_store = OnDemandTargetStoreV2()

    # Load S2 and S3 in chunks to conserve RAM
    total_targets_loaded = 0
    for file_path in [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"]:
        print(f"Reading {file_path.name}...", flush=True)
        for chunk_df in pd.read_csv(file_path, sep="\t", chunksize=500000, dtype=str):
            chunk_df = chunk_df.fillna("")
            recs = chunk_df.to_dict("records")
            blocker.build_index(recs)

            for rec in recs:
                mid = rec["entity_id"]
                c = normalize_country(rec.get("country", ""))
                nn = normalize_name(rec.get("business_name", ""))
                na = normalize_address(rec.get("business_address", ""))
                target_store.add_target_raw(mid, c, nn, na)

            total_targets_loaded += len(recs)
            pid, ram, cpu = get_telemetry()
            print(f"Loaded {total_targets_loaded:,} targets into Blocker Index | RAM: {ram:.2f}GB | PID: {pid}", flush=True)
            gc.collect()

    print(f"Target index built in {time.time() - t0_block:.2f}s. Total targets indexed: {len(target_store):,}", flush=True)

    # Step 3: Initialize Output TSV Files
    print("\n[STAGE 3/4] Initializing output files...", flush=True)
    with open(MATCHING_OUT, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Step 4: Stream Test Source 1 in Batches & Run Ensemble Inference
    print("\n[STAGE 4/4] Processing S1 test records in batches...", flush=True)
    total_s1_processed = 0
    total_candidate_pairs = 0
    total_matched_s1 = 0
    total_unmatched_s1 = 0
    total_final_matches = 0

    s1_batch_size = 25000
    last_print = time.time()
    TOTAL_S1 = 1732544

    for s1_chunk_df in pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", chunksize=s1_batch_size, dtype=str):
        s1_chunk_df = s1_chunk_df.fillna("")
        s1_recs = s1_chunk_df.to_dict("records")

        match_lines = []
        cand_lines = []

        # Gather batch candidate pairs
        batch_X_list = []
        batch_meta = [] # list of (s1_id, cands_list, start_idx, end_idx)
        curr_row = 0

        for s1_rec in s1_recs:
            s1_id = s1_rec["entity_id"]
            c = s1_rec.get("country", "")
            rn = s1_rec.get("business_name", "")
            ra = s1_rec.get("business_address", "")

            cands = blocker.get_candidates(s1_rec)
            total_candidate_pairs += len(cands)

            s1_prep = precompute_s1_v2(s1_id, c, rn, ra)

            if cands:
                cands_list = list(cands)
                X_s1 = np.zeros((len(cands_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
                for r_idx, tid in enumerate(cands_list):
                    fill_features_v2(X_s1, r_idx, s1_prep, target_store, tid)

                batch_X_list.append(X_s1)
                n_rows = len(cands_list)
                batch_meta.append((s1_id, cands_list, curr_row, curr_row + n_rows))
                curr_row += n_rows
            else:
                batch_meta.append((s1_id, [], 0, 0))

        # Vectorized scoring for batch
        if batch_X_list:
            X_batch = np.vstack(batch_X_list)
            p_lgb = lgb_model.predict_proba(X_batch)[:, 1]
            p_xgb = xgb_model.predict_proba(X_batch)[:, 1]
            p_ens = 0.5 * p_lgb + 0.5 * p_xgb

            for s1_id, cands_list, start_i, end_i in batch_meta:
                cand_str = ",".join(cands_list) if cands_list else ""
                cand_lines.append(f"{s1_id}\t{cand_str}\n")

                if cands_list:
                    probs = p_ens[start_i:end_i]
                    matched_ids = [cands_list[i] for i, prob in enumerate(probs) if prob >= 0.995]
                    match_str = ",".join(matched_ids) if matched_ids else ""
                    match_lines.append(f"{s1_id}\t{match_str}\n")

                    if matched_ids:
                        total_matched_s1 += 1
                        total_final_matches += len(matched_ids)
                    else:
                        total_unmatched_s1 += 1
                else:
                    match_lines.append(f"{s1_id}\t\n")
                    total_unmatched_s1 += 1
        else:
            for s1_id, _, _, _ in batch_meta:
                cand_lines.append(f"{s1_id}\t\n")
                match_lines.append(f"{s1_id}\t\n")
                total_unmatched_s1 += 1

        # Append to files
        with open(MATCHING_OUT, "a", encoding="utf-8") as f_match:
            f_match.writelines(match_lines)

        with open(CANDIDATE_OUT, "a", encoding="utf-8") as f_cand:
            f_cand.writelines(cand_lines)

        total_s1_processed += len(s1_recs)

        # Telemetry update
        elapsed = time.time() - start_time
        rate = total_s1_processed / elapsed if elapsed > 0 else 0
        eta = (TOTAL_S1 - total_s1_processed) / rate if rate > 0 else 0
        pid, ram, cpu = get_telemetry()
        cands_per_s1 = total_candidate_pairs / total_s1_processed if total_s1_processed > 0 else 0

        print(f"[FULL TEST INFERENCE] S1: {total_s1_processed:,} / {TOTAL_S1:,} ({(total_s1_processed/TOTAL_S1)*100:.2f}%) | "
              f"Candidates: {total_candidate_pairs:,} ({cands_per_s1:.1f}/S1) | "
              f"Matched S1: {total_matched_s1:,} (Unmatched: {total_unmatched_s1:,}) | "
              f"Matches Found: {total_final_matches:,} | "
              f"Rate: {rate:.1f} S1/sec | Elapsed: {elapsed/60:.2f}m | ETA: {eta/60:.2f}m | "
              f"RAM: {ram:.2f}GB | CPU: {cpu}% | PID: {pid}", flush=True)

        gc.collect()

    print("\n" + "=" * 80, flush=True)
    print("FULL INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total S1 Processed    : {total_s1_processed:,}")
    print(f"Total Candidate Pairs : {total_candidate_pairs:,} ({total_candidate_pairs/total_s1_processed:.1f} cands/S1)")
    print(f"Total Final Matches   : {total_final_matches:,}")
    print(f"Matched S1 Count      : {total_matched_s1:,} ({(total_matched_s1/total_s1_processed)*100:.2f}%)")
    print(f"Unmatched S1 Count    : {total_unmatched_s1:,} ({(total_unmatched_s1/total_s1_processed)*100:.2f}%)")
    print(f"Total Elapsed Time    : {time.time() - start_time:.2f} s ({(time.time() - start_time)/60:.2f} min)")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
