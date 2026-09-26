"""
Validation Script: 100-S1 Test Blocker Pruning Comparison
Goal: Compare CURRENT production blocker vs EXPERIMENTAL 10,000-limit blocker
on the REAL test target pool (9.97M records) and 100 S1 records.

DO NOT MODIFY PRODUCTION FILES.
DO NOT RUN LIGHTGBM.
DO NOT RUN FULL INFERENCE.
DO NOT GENERATE SUBMISSION FILES.
"""

import os
import sys
import time
import copy
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Path setup
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from blocking import EntityBlocker, extract_3grams

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 80, flush=True)
print("=== AMAZON ML CHALLENGE — 100-S1 TEST BLOCKER COMPARISON ===", flush=True)
print("=" * 80, flush=True)
print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

# 1. LOAD REAL TEST TARGET POOL (S2 + S3)
test_dir = base_dir / "dataset" / "test"
s2_path = test_dir / "test_source2.tsv"
s3_path = test_dir / "test_source3.tsv"
s1_path = test_dir / "test_source1.tsv"

print("--- 1. BUILDING FULL TEST BLOCKER INDEX (S2 + S3 TARGET POOL) ---", flush=True)
t0_index = time.time()
blocker_current = EntityBlocker()

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
            norm_n = normalize_name(str(r_n) if pd.notna(r_n) else "")
            norm_a = normalize_address(str(r_a) if pd.notna(r_a) else "")
            blocker_current.add_target_record(mid, c, norm_n, norm_a)
            total_target_loaded += 1
            
            if total_target_loaded - last_report_count >= 100000:
                last_report_count = total_target_loaded
                elapsed = time.time() - t0_index
                print(f"  Indexed {total_target_loaded:,} target records... ({elapsed:.2f}s elapsed, RAM: {get_ram_mb():.1f} MB)", flush=True)

t_index = time.time() - t0_index
print(f"Full Target Index built in {t_index:.2f}s for {total_target_loaded:,} records. (RAM: {get_ram_mb():.1f} MB)\n", flush=True)

# 2. CONSTRUCT EXPERIMENTAL 10,000-LIMIT BLOCKER
print("--- 2. CONSTRUCTING EXPERIMENTAL 10,000-LIMIT BLOCKER ---", flush=True)
t0_prune = time.time()
blocker_10k = copy.deepcopy(blocker_current)

pruned_keys_cnt = 0
MAX_LIMIT = 10000

channels_10k = [
    blocker_10k.name_pref2_idx,
    blocker_10k.addr_pref2_idx,
    blocker_10k.name_rare_tok_idx,
    blocker_10k.addr_rare_tok_idx,
    blocker_10k.name_ngram_idx
]

for ch in channels_10k:
    to_remove = [k for k, s in ch.items() if len(s) > MAX_LIMIT]
    for k in to_remove:
        del ch[k]
        pruned_keys_cnt += 1

t_prune = time.time() - t0_prune
print(f"Pruning complete in {t_prune:.3f}s. Total keys pruned with size > 10,000: {pruned_keys_cnt:,} keys.\n", flush=True)

# 3. LOAD 100 S1 TEST RECORDS
print("--- 3. EVALUATING CANDIDATE GENERATION ON 100 S1 TEST RECORDS ---", flush=True)
num_s1 = 100
s1_df_100 = pd.read_csv(s1_path, sep="\t", nrows=num_s1, dtype=str)

# Evaluation - CURRENT BLOCKER
t0_curr_cand = time.time()
curr_cands_total = 0
curr_counts_per_s1 = []
curr_cand_sets = {}

for _, r in s1_df_100.iterrows():
    s1_id = r['entity_id']
    raw_rec = {
        'entity_id': s1_id,
        'business_name': normalize_name(str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""),
        'business_address': normalize_address(str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""),
        'country': str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    }
    cands = blocker_current.get_candidates(raw_rec)
    curr_cands_total += len(cands)
    curr_counts_per_s1.append(len(cands))
    curr_cand_sets[s1_id] = cands

t_curr_cand = time.time() - t0_curr_cand

# Evaluation - 10,000-LIMIT BLOCKER
t0_10k_cand = time.time()
pruned_cands_total = 0
pruned_counts_per_s1 = []
pruned_cand_sets = {}

for _, r in s1_df_100.iterrows():
    s1_id = r['entity_id']
    raw_rec = {
        'entity_id': s1_id,
        'business_name': normalize_name(str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""),
        'business_address': normalize_address(str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""),
        'country': str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    }
    cands = blocker_10k.get_candidates(raw_rec)
    pruned_cands_total += len(cands)
    pruned_counts_per_s1.append(len(cands))
    pruned_cand_sets[s1_id] = cands

t_10k_cand = time.time() - t0_10k_cand

# Calculate statistics
curr_avg = curr_cands_total / num_s1
pruned_avg = pruned_cands_total / num_s1

abs_reduction = curr_cands_total - pruned_cands_total
pct_reduction = (abs_reduction / curr_cands_total) * 100.0 if curr_cands_total > 0 else 0.0

peak_ram = get_ram_mb()

# Inspect removed candidates
total_removed_pairs = 0
removed_sample = []

for s1_id in curr_cand_sets:
    diff = curr_cand_sets[s1_id] - pruned_cand_sets[s1_id]
    total_removed_pairs += len(diff)
    if diff and len(removed_sample) < 5:
        removed_sample.append((s1_id, len(diff), sorted(list(diff))[:3]))

print(f"Current Blocker:    Total Candidates = {curr_cands_total:,} | Avg/S1 = {curr_avg:,.1f} | Min = {min(curr_counts_per_s1):,} | Max = {max(curr_counts_per_s1):,} | Time = {t_curr_cand:.3f}s", flush=True)
print(f"10,000-Limit:       Total Candidates = {pruned_cands_total:,} | Avg/S1 = {pruned_avg:,.1f} | Min = {min(pruned_counts_per_s1):,} | Max = {max(pruned_counts_per_s1):,} | Time = {t_10k_cand:.3f}s", flush=True)
print(f"Candidate Reduction: {abs_reduction:,} pairs removed ({pct_reduction:.2f}% reduction)", flush=True)
print(f"Sample of Removed Candidate Batches:", flush=True)
for sid, rcnt, rlist in removed_sample:
    print(f"  S1 {sid}: removed {rcnt:,} candidate IDs (e.g. {rlist})", flush=True)

# 4. REQUIRED FINAL OUTPUT
print("\n" + "=" * 60, flush=True)
print("=== 100-S1 TEST BLOCKER COMPARISON ===", flush=True)
print("=" * 60, flush=True)
print("Current blocker:", flush=True)
print(f"  candidates = {curr_cands_total:,}", flush=True)
print(f"  avg/S1 = {curr_avg:,.1f}", flush=True)
print(flush=True)
print("10,000-limit blocker:", flush=True)
print(f"  candidates = {pruned_cands_total:,}", flush=True)
print(f"  avg/S1 = {pruned_avg:,.1f}", flush=True)
print(flush=True)
print("Reduction:", flush=True)
print(f"  {pct_reduction:.2f} % ({abs_reduction:,} candidate pairs removed)", flush=True)
print(flush=True)
print("Peak RAM:", flush=True)
print(f"  {peak_ram:.1f} MB", flush=True)
print(flush=True)
print("Runtime:", flush=True)
print(f"  Index build: {t_index:.2f} s | Current cand gen: {t_curr_cand:.3f} s | 10k cand gen: {t_10k_cand:.3f} s", flush=True)
print("=" * 60 + "\n", flush=True)

print("To re-run this validation script at any time, execute:", flush=True)
print(r"py -3 scratch\validate_pruning_on_test_sample.py", flush=True)
