"""
Profiling Script: Feature-by-Feature Execution Time Profile
Measures time spent on each individual feature (0 to 12) across 100,000 candidate pairs.
"""

import os
import sys
import time
import psutil
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Path setup
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from blocking import EntityBlocker
from inference_optimized import (
    OnDemandTargetStore,
    precompute_s1_record,
    fast_norm_name,
    fast_norm_addr
)

test_dir = base_dir / "dataset" / "test"
s2_path = test_dir / "test_source2.tsv"
s3_path = test_dir / "test_source3.tsv"
s1_path = test_dir / "test_source1.tsv"

print("=" * 80)
print("=== FEATURE EXTRACTION PROFILING ===")
print("=" * 80)

# Build Target Store & Blocker on 500k sample for fast profiling
print("Building Target Store & Blocker sample...")
target_store = OnDemandTargetStore()
blocker = EntityBlocker()

target_loaded = 0
for fpath in [s2_path, s3_path]:
    if target_loaded >= 500000: break
    for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
        if target_loaded >= 500000: break
        sub = chunk.iloc[:min(len(chunk), 500000 - target_loaded)]
        for mid, r_n, r_a, c in zip(sub['entity_id'], sub['business_name'], sub['business_address'], sub['country'].fillna('UNKNOWN')):
            nn = fast_norm_name(r_n)
            na = fast_norm_addr(r_a)
            target_store.add_target_raw(mid, str(c), nn, na)
            blocker.add_target_record(mid, str(c), nn, na)
            target_loaded += 1

# Prune with 10k limit
for ch in [blocker.name_pref2_idx, blocker.addr_pref2_idx, blocker.name_rare_tok_idx, blocker.addr_rare_tok_idx, blocker.name_ngram_idx]:
    for k in [k for k, s in ch.items() if len(s) > 10000]:
        del ch[k]

# Collect 100,000 candidate pairs
s1_df = pd.read_csv(s1_path, sep="\t", nrows=50, dtype=str)
pair_data = []

for _, r in s1_df.iterrows():
    s1_id = r['entity_id']
    c = str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    s1_rec = precompute_s1_record(s1_id, c, r.get('business_name', ''), r.get('business_address', ''))
    cands = blocker.get_candidates(s1_rec['raw_rec'])
    for tid in cands:
        if tid in target_store.raw_attr:
            pair_data.append((s1_rec, tid))
        if len(pair_data) >= 100000:
            break
    if len(pair_data) >= 100000:
        break

print(f"Collected {len(pair_data):,} candidate pairs for profiling.\n")

# Pre-parse target tuples for fairness
parsed_pairs = []
for s1, tid in pair_data:
    t_parsed = target_store.get_parsed(tid)
    parsed_pairs.append((s1, t_parsed, tid))

# Profile individual feature components
n_pairs = len(parsed_pairs)

# Feature 0: Name exact match
t0 = time.time()
for s1, t, tid in parsed_pairs:
    f0 = 1.0 if s1['norm_n'] == t[1] and s1['norm_n'] != "" else 0.0
t_f0 = time.time() - t0

# Feature 1 & 9: Name token Jaccard & overlap
t0 = time.time()
for s1, t, tid in parsed_pairs:
    l1, l2 = s1['len_toks_n'], t[10]
    if l1 > 0 and l2 > 0:
        ov = len(s1['toks_n'] & t[3])
        jac = ov / (l1 + l2 - ov)
    else: ov, jac = 0, 0.0
t_f1_9 = time.time() - t0

# Feature 2: Name 3-gram char Jaccard
t0 = time.time()
for s1, t, tid in parsed_pairs:
    l1, l2 = s1['len_ng_n'], t[12]
    if l1 > 0 and l2 > 0:
        ov = len(s1['ngrams_n'] & t[5])
        jac = ov / (l1 + l2 - ov)
    else: jac = 0.0
t_f2 = time.time() - t0

# Feature 3 & 10: Address token Jaccard & overlap
t0 = time.time()
for s1, t, tid in parsed_pairs:
    l1, l2 = s1['len_toks_a'], t[11]
    if l1 > 0 and l2 > 0:
        ov = len(s1['toks_a'] & t[4])
        jac = ov / (l1 + l2 - ov)
    else: ov, jac = 0, 0.0
t_f3_10 = time.time() - t0

# Feature 4: Address 3-gram char Jaccard
t0 = time.time()
for s1, t, tid in parsed_pairs:
    l1, l2 = s1['len_ng_a'], t[13]
    if l1 > 0 and l2 > 0:
        ov = len(s1['ngrams_a'] & t[6])
        jac = ov / (l1 + l2 - ov)
    else: jac = 0.0
t_f4 = time.time() - t0

# Feature 5: Address digit Jaccard
t0 = time.time()
for s1, t, tid in parsed_pairs:
    l1, l2 = s1['len_dig_a'], t[14]
    if l1 > 0 and l2 > 0:
        ov = len(s1['digits_a'] & t[7])
        jac = ov / (l1 + l2 - ov)
    else: jac = 0.5
t_f5 = time.time() - t0

# Feature 6, 7, 8, 11, 12: Country match, length diffs, missing addr, is_s2
t0 = time.time()
for s1, t, tid in parsed_pairs:
    cm = 1.0 if s1['c'] == t[0] and s1['c'] != "" else 0.0
    ld_n = float(abs(s1['len_n'] - t[8]))
    ld_a = float(abs(s1['len_a'] - t[9]))
    ma = 1.0 if s1['len_a'] == 0 or t[9] == 0 else 0.0
    s2 = t[15]
t_f_misc = time.time() - t0

total_prof = t_f0 + t_f1_9 + t_f2 + t_f3_10 + t_f4 + t_f5 + t_f_misc

print(f"Profiling Results on {n_pairs:,} Candidate Pairs:")
print(f"  Name Token Jaccard (F1 & F9):     {t_f1_9:.4f} s ({t_f1_9/total_prof*100:.1f}%)")
print(f"  Name 3-Gram Jaccard (F2):         {t_f2:.4f} s ({t_f2/total_prof*100:.1f}%)")
print(f"  Addr Token Jaccard (F3 & F10):    {t_f3_10:.4f} s ({t_f3_10/total_prof*100:.1f}%)")
print(f"  Addr 3-Gram Jaccard (F4):         {t_f4:.4f} s ({t_f4/total_prof*100:.1f}%)")
print(f"  Addr Digit Jaccard (F5):          {t_f5:.4f} s ({t_f5/total_prof*100:.1f}%)")
print(f"  Name Exact Match (F0):            {t_f0:.4f} s ({t_f0/total_prof*100:.1f}%)")
print(f"  Misc (F6, F7, F8, F11, F12):      {t_f_misc:.4f} s ({t_f_misc/total_prof*100:.1f}%)")
print(f"  TOTAL PROFILED TIME:              {total_prof:.4f} s\n")
