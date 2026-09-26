"""
Diagnostic Benchmark: 100-S1 End-to-End Inference Benchmark with 10,000 Blocker Pruning Limit.
Goal: Obtain an accurate runtime estimate for full submission on 1,732,544 S1 records.

DO NOT MODIFY PRODUCTION FILES PERMANENTLY.
DO NOT RUN ALL 1,732,544 S1 RECORDS.
DO NOT GENERATE SUBMISSION FILES.
"""

import os
import sys
import time
import psutil
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Path setup
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from blocking import EntityBlocker, extract_3grams
from inference_optimized import (
    OnDemandTargetStore,
    precompute_s1_record,
    fill_features_optimized,
    fast_norm_name,
    fast_norm_addr
)

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 80, flush=True)
print("=== 100-S1 END-TO-END INFERENCE BENCHMARK (10,000 PRUNING LIMIT) ===", flush=True)
print("=" * 80, flush=True)
print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

test_dir = base_dir / "dataset" / "test"
model_path = base_dir / "models" / "lightgbm_model.joblib"

# 1. LOAD MODEL
print("--- 1. LOADING LIGHTGBM MODEL ---", flush=True)
t0_model = time.time()
model = joblib.load(model_path)
t_model = time.time() - t0_model
print(f"Loaded LightGBM model in {t_model:.3f}s (RAM: {get_ram_mb():.1f} MB)\n", flush=True)

# 2. LOAD TARGET POOL & BUILD 10k PRUNED BLOCKER
print("--- 2. BUILDING TARGET FEATURE STORE & 10,000-LIMIT BLOCKER INDEX ---", flush=True)
t0_index = time.time()

target_store = OnDemandTargetStore()
blocker = EntityBlocker()

s2_path = test_dir / "test_source2.tsv"
s3_path = test_dir / "test_source3.tsv"
s1_path = test_dir / "test_source1.tsv"

target_files = [
    (s2_path, "Source-2"),
    (s3_path, "Source-3")
]

total_target_loaded = 0
last_report_count = 0

for fpath, label in target_files:
    print(f"Loading and indexing target pool from {label} ({fpath.name})...", flush=True)
    for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
        eids = chunk['entity_id'].values
        raw_names = chunk['business_name'].values
        raw_addrs = chunk['business_address'].values
        countries = chunk['country'].fillna('UNKNOWN').astype(str).values
        
        for mid, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
            norm_n = fast_norm_name(r_n)
            norm_a = fast_norm_addr(r_a)
            target_store.add_target_raw(mid, c, norm_n, norm_a)
            blocker.add_target_record(mid, c, norm_n, norm_a)
            total_target_loaded += 1
            
            if total_target_loaded - last_report_count >= 100000:
                last_report_count = total_target_loaded
                elapsed = time.time() - t0_index
                print(f"  Indexed {total_target_loaded:,} target records... ({elapsed:.2f}s elapsed, RAM: {get_ram_mb():.1f} MB)", flush=True)

# Apply 10,000 posting-list pruning cap across all channels
MAX_LIMIT = 10000
pruned_keys_cnt = 0
channels = [
    blocker.name_pref2_idx,
    blocker.addr_pref2_idx,
    blocker.name_rare_tok_idx,
    blocker.addr_rare_tok_idx,
    blocker.name_ngram_idx
]

for ch in channels:
    to_remove = [k for k, s in ch.items() if len(s) > MAX_LIMIT]
    for k in to_remove:
        del ch[k]
        pruned_keys_cnt += 1

t_index = time.time() - t0_index
print(f"Target index build & 10,000-limit pruning complete in {t_index:.2f}s. Total keys pruned: {pruned_keys_cnt:,} (RAM: {get_ram_mb():.1f} MB)\n", flush=True)

# 3. RUN 100-S1 END-TO-END INFERENCE
print("--- 3. RUNNING 100-S1 END-TO-END INFERENCE BENCHMARK ---", flush=True)
num_s1_test = 100

s1_df_100 = pd.read_csv(s1_path, sep="\t", nrows=num_s1_test, dtype=str)
eids = s1_df_100['entity_id'].values
raw_names = s1_df_100['business_name'].values
raw_addrs = s1_df_100['business_address'].values
countries = s1_df_100['country'].fillna('UNKNOWN').astype(str).values

t0_inference = time.time()

t_cand_gen = 0.0
t_feat_ext = 0.0
t_predict = 0.0

total_candidate_pairs = 0
total_predicted_matches = 0

s1_precomputed = []
pair_meta = []

# A. Candidate Generation (with progress every 10 S1)
t0_cand = time.time()
for idx, (s1_id, r_n, r_a, c) in enumerate(zip(eids, raw_names, raw_addrs, countries), 1):
    s1_rec = precompute_s1_record(s1_id, c, r_n, r_a)
    s1_precomputed.append(s1_rec)
    
    cands = blocker.get_candidates(s1_rec['raw_rec'])
    total_candidate_pairs += len(cands)
    
    for tid in cands:
        if tid in target_store.raw_attr:
            pair_meta.append((len(s1_precomputed) - 1, tid, s1_id))
            
    if idx % 10 == 0 or idx == num_s1_test:
        elapsed = time.time() - t0_cand
        print(f"  [Progress] S1 {idx:3d}/{num_s1_test} processed | Candidates generated: {total_candidate_pairs:,} ({elapsed:.2f}s, RAM: {get_ram_mb():.1f} MB)", flush=True)

t_cand_gen = time.time() - t0_cand

# B. Feature Extraction directly into NumPy Float32 Matrix
if pair_meta:
    t0_feat = time.time()
    X_batch = np.empty((len(pair_meta), 13), dtype=np.float32)
    for p_idx, (s1_idx, tid, _) in enumerate(pair_meta):
        fill_features_optimized(X_batch, p_idx, s1_precomputed[s1_idx], target_store, tid)
    t_feat_ext = time.time() - t0_feat
    
    # C. LightGBM Scoring
    t0_pred = time.time()
    probs = model.predict_proba(X_batch)[:, 1]
    t_predict = time.time() - t0_pred
    
    match_pairs = (probs >= 0.98).sum()
    total_predicted_matches = int(match_pairs)
    
t_infer_total = time.time() - t0_inference
total_pipeline_time = t_index + t_infer_total
peak_ram = get_ram_mb()

# 4. EXTRAPOLATION & FULL SUBMISSION ESTIMATES
total_full_s1 = 1732544
s1_per_sec = num_s1_test / t_infer_total if t_infer_total > 0 else 0
est_full_infer_sec = total_full_s1 / s1_per_sec if s1_per_sec > 0 else 0
est_full_infer_min = est_full_infer_sec / 60.0
est_full_infer_hours = est_full_infer_sec / 3600.0

# Output writing + validation overhead estimate on disk (approx 90 seconds)
est_output_writing_sec = 90.0
est_total_submission_sec = t_index + est_full_infer_sec + est_output_writing_sec
est_total_submission_min = est_total_submission_sec / 60.0
est_total_submission_hours = est_total_submission_sec / 3600.0

print("\n" + "=" * 80, flush=True)
print("=== 100-S1 END-TO-END INFERENCE BENCHMARK METRICS ===", flush=True)
print("=" * 80, flush=True)
print(f"1. Target Index Build Time:     {t_index:.2f} s ({t_index/60.0:.2f} min)", flush=True)
print(f"2. Candidate Generation Time:   {t_cand_gen:.3f} s", flush=True)
print(f"3. Feature Extraction Time:     {t_feat_ext:.3f} s", flush=True)
print(f"4. LightGBM Prediction Time:    {t_predict:.3f} s", flush=True)
print(f"5. Total 100-S1 Test Time:      {total_pipeline_time:.2f} s", flush=True)
print(f"6. Peak RAM Usage:              {peak_ram:.1f} MB", flush=True)
print(f"7. Candidate Pairs Generated:   {total_candidate_pairs:,}", flush=True)
print(f"8. Predicted Matches (T>=0.98): {total_predicted_matches:,}", flush=True)
print(f"9. S1 Processing Throughput:    {s1_per_sec:,.2f} S1/sec", flush=True)
print(f"10. Est. Full 1.73M Inference:  {est_full_infer_sec:.1f} s ({est_full_infer_min:.1f} min / {est_full_infer_hours:.2f} hours)", flush=True)
print(f"11. Est. Total Submission Time: {est_total_submission_sec:.1f} s ({est_total_submission_min:.1f} min / {est_total_submission_hours:.2f} hours)", flush=True)
print("=" * 80 + "\n", flush=True)

# Automatic Check against 2-hour threshold
within_2_hours = est_total_submission_hours <= 2.0

if within_2_hours:
    print(f"FEASIBILITY CHECK: PASS — Estimated total submission time of {est_total_submission_min:.1f} minutes ({est_total_submission_hours:.2f} hours) is WITHIN the 2-hour limit!", flush=True)
else:
    print(f"FEASIBILITY CHECK: FAIL — Estimated total submission time of {est_total_submission_hours:.2f} hours EXCEEDS the 2-hour limit!", flush=True)
    bottleneck = "Index building" if t_index > est_full_infer_sec else "Feature extraction & LightGBM scoring"
    print(f"Primary Bottleneck: {bottleneck}", flush=True)

print("=" * 80 + "\n", flush=True)
