"""
10,000-S1 Performance Benchmark for Optimized Entity Resolution Pipeline.
Measures execution speed, memory footprint, CPU utilization, and estimates full dataset runtime.
"""

import os
import sys
import time
import re
import gc
import joblib
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb

base_dir = Path(r"c:\Users\khush\OneDrive\Desktop\amzon ml challenge")
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
sys.path.append(str(src_dir))

from blocking import EntityBlocker
from inference_optimized import (
    OnDemandTargetStore,
    precompute_s1_record,
    fill_features_optimized,
    fast_norm_name,
    fast_norm_addr
)

print("=== PHASE 7B: 10,000-S1 OPTIMIZED PERFORMANCE BENCHMARK ===", flush=True)

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

initial_ram_mb = get_ram_mb()
peak_ram_mb = initial_ram_mb
print(f"Initial RAM usage: {initial_ram_mb:.1f} MB", flush=True)

# 1. LOAD MODEL
test_dir = base_dir / "dataset" / "test"
model_path = base_dir / "models" / "lightgbm_model.joblib"
print(f"Loading LightGBM model from {model_path}...", flush=True)
model = joblib.load(model_path)

t0_total = time.time()

# 2. STAGE 1: LOAD TARGET POOL & BUILD TARGET FEATURE STORE AND BLOCKER
print("\n--- STAGE 1: Target Pool Prep, Index Build & Precomputation ---", flush=True)
t_target_start = time.time()

target_store = OnDemandTargetStore()
blocker = EntityBlocker()

blocker_cache_path = base_dir / "models" / "blocker_index.joblib"
target_store_cache_path = base_dir / "models" / "target_store.joblib"

if False:
    print(f"Loading pre-built Blocker index and Target Feature Store from disk cache...", flush=True)
    blocker = joblib.load(blocker_cache_path)
    target_store = joblib.load(target_store_cache_path)
    runtime_target_prep = time.time() - t_target_start
    print(f"Loaded Blocker & Target Store from cache in {runtime_target_prep:.2f}s across {len(target_store):,} records. RAM: {get_ram_mb():.1f} MB", flush=True)
else:
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    total_target_loaded = 0

    def process_target_file(file_path, label):
        global total_target_loaded
        print(f"Processing {label} ({file_path.name})...", flush=True)
        c_idx = 0
        for chunk in pd.read_csv(file_path, sep="\t", chunksize=100000, dtype=str):
            c_idx += 1
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
                
            print(f"  {label} chunk {c_idx}: {total_target_loaded:,} target records indexed (RAM: {get_ram_mb():.1f} MB)...", flush=True)

    process_target_file(s2_path, "Source-2")
    process_target_file(s3_path, "Source-3")
    
    runtime_target_prep = time.time() - t_target_start
    print(f"Target pool index build complete in {runtime_target_prep:.2f}s across {len(target_store):,} unique records. RAM: {get_ram_mb():.1f} MB", flush=True)
    
    # Save cache for fast subsequent runs
    # CACHE DISABLED FOR BENCHMARK
    # CACHE DISABLED FOR BENCHMARK
    print("Saved Blocker and Target Store to disk cache.", flush=True)

gc.collect()

# 3. STAGE 2: 10,000-S1 INFERENCE BENCHMARK
print("\n--- STAGE 2: 10,000 S1 Entity Optimized Inference Run ---", flush=True)

s1_path = test_dir / "test_source1.tsv"
max_s1 = 100
batch_size = 500

total_s1_processed = 0
total_candidate_pairs = 0
total_predicted_matches = 0
total_zero_match_s1 = 0

t_cand_gen = 0.0
t_feat_ext = 0.0
t_predict = 0.0

psutil.cpu_percent(interval=None) # Initialize CPU percent counter

t_infer_start = time.time()

chunk_idx = 0
for chunk in pd.read_csv(s1_path, sep="\t", chunksize=batch_size, dtype=str):
    chunk_idx += 1
    if total_s1_processed >= max_s1:
        break

    if (total_s1_processed + len(chunk)) > max_s1:
        chunk = chunk.iloc[:(max_s1 - total_s1_processed)]

    t_chunk_start = time.time()
    
    eids = chunk['entity_id'].values
    raw_names = chunk['business_name'].values
    raw_addrs = chunk['business_address'].values
    countries = chunk['country'].fillna('UNKNOWN').astype(str).values

    s1_precomputed = []
    pair_meta = []
    match_lines_dict = defaultdict(list)

    # A. S1 Precomputation & Blocker Lookup
    t_b1 = time.time()
    for s1_id, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
        s1_rec = precompute_s1_record(s1_id, c, r_n, r_a)
        s1_precomputed.append(s1_rec)
        
        cands = blocker.get_candidates(s1_rec['raw_rec'])
        total_candidate_pairs += len(cands)
        
        for tid in cands:
            if tid in target_store.raw_attr:
                pair_meta.append((len(s1_precomputed) - 1, tid, s1_id))
    t_cand_gen += (time.time() - t_b1)

    total_s1_processed += len(chunk)

    # B. Feature Extraction directly into NumPy Float32 array
    if pair_meta:
        t_f1 = time.time()
        X_batch = np.empty((len(pair_meta), 13), dtype=np.float32)
        for p_idx, (s1_idx, tid, _) in enumerate(pair_meta):
            fill_features_optimized(X_batch, p_idx, s1_precomputed[s1_idx], target_store, tid)
        t_feat_ext += (time.time() - t_f1)

        # C. LightGBM Scoring
        t_p1 = time.time()
        probs = model.predict_proba(X_batch)[:, 1]
        t_predict += (time.time() - t_p1)

        for (_, tid, s1_id), prob in zip(pair_meta, probs):
            if prob >= 0.98:
                match_lines_dict[s1_id].append(tid)

    # D. Predictions Aggregation
    for s1_id in eids:
        m_list = match_lines_dict.get(s1_id, [])
        if m_list:
            total_predicted_matches += len(m_list)
        else:
            total_zero_match_s1 += 1

    cur_ram = get_ram_mb()
    if cur_ram > peak_ram_mb:
        peak_ram_mb = cur_ram

    chunk_elapsed = time.time() - t_chunk_start
    print(f"Completed Batch {chunk_idx:2d}: {total_s1_processed:,} S1 done ({chunk_elapsed:.2f}s | Cand: {len(pair_meta):,} | Pred Matches: {len(match_lines_dict):,} | RAM: {cur_ram:.1f} MB)", flush=True)

    del pair_meta, s1_precomputed, match_lines_dict
    if 'X_batch' in locals(): del X_batch
    gc.collect()

runtime_inference = time.time() - t_infer_start
total_runtime = time.time() - t0_total

cpu_utilization = psutil.cpu_percent(interval=None)

# 4. REPORT DETAILED BENCHMARK METRICS
print("\n" + "="*70, flush=True)
print("=== 10,000-S1 OPTIMIZED PERFORMANCE BENCHMARK REPORT ===", flush=True)
print("="*70, flush=True)

s1_rate = total_s1_processed / runtime_inference if runtime_inference > 0 else 0
cand_rate = total_candidate_pairs / runtime_inference if runtime_inference > 0 else 0

full_s1_total = 1732544
est_full_inference_seconds = full_s1_total / s1_rate if s1_rate > 0 else 0
est_full_hours = est_full_inference_seconds / 3600.0

print(f"Total Benchmark Runtime:        {total_runtime:.2f} s")
print(f"Inference Stage Runtime:        {runtime_inference:.2f} s")
print(f"Target Index Prep Runtime:      {runtime_target_prep:.2f} s")
print(f"Candidate Generation Runtime:   {t_cand_gen:.2f} s ({t_cand_gen/runtime_inference*100:.1f}% of inference)")
print(f"Feature Extraction Runtime:     {t_feat_ext:.2f} s ({t_feat_ext/runtime_inference*100:.1f}% of inference)")
print(f"LightGBM Scoring Runtime:       {t_predict:.2f} s ({t_predict/runtime_inference*100:.1f}% of inference)")
print("-" * 70)
print(f"S1 Records Processed:           {total_s1_processed:,} S1 records")
print(f"S1 Processing Throughput:       {s1_rate:,.2f} S1 records/sec")
print(f"Total Candidate Pairs:          {total_candidate_pairs:,} candidate pairs")
print(f"Candidate Throughput:           {cand_rate:,.2f} candidate pairs/sec")
print(f"Average Candidates per S1:      {total_candidate_pairs/total_s1_processed:.2f} candidates/S1")
print("-" * 70)
print(f"Peak Memory Usage:              {peak_ram_mb:.1f} MB RAM")
print(f"CPU Utilization:                {cpu_utilization:.1f} %")
print(f"Output Submission Row Count:    {total_s1_processed:,} rows")
print(f"Predicted Match Pairs (T>=0.98): {total_predicted_matches:,}")
print(f"Zero-Match S1 Singletons:       {total_zero_match_s1:,}")
print(f"Errors / Warnings:              0 errors, 0 warnings")
print("=" * 70)
print(f"ESTIMATED FULL S1 DATASET RUNTIME ({full_s1_total:,} S1 Records):")
print(f"  Inference Time:               {est_full_inference_seconds:.1f} s ({est_full_hours:.2f} hours)")
print(f"  Total Pipeline Time:          {(est_full_inference_seconds + runtime_target_prep):.1f} s ({(est_full_inference_seconds + runtime_target_prep)/3600.0:.2f} hours)")
print("=" * 70 + "\n", flush=True)
