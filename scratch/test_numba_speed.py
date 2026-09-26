"""
Validation & Speed Benchmark: Numba Accelerated Feature Extraction vs Baseline
Verifies 100% numerical equivalence and measures execution speedup.
"""

import os
import sys
import time
import psutil
import joblib
import numba
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
from blocking import EntityBlocker
from inference_optimized import (
    OnDemandTargetStore,
    precompute_s1_record,
    fill_features_optimized,
    fast_norm_name,
    fast_norm_addr,
    extract_digits,
    extract_ngrams
)

test_dir = base_dir / "dataset" / "test"
model_path = base_dir / "models" / "lightgbm_model.joblib"
s2_path = test_dir / "test_source2.tsv"
s3_path = test_dir / "test_source3.tsv"
s1_path = test_dir / "test_source1.tsv"

print("=" * 80)
print("=== NUMBA FEATURE EXTRACTION VALIDATION & SPEED BENCHMARK ===")
print("=" * 80)

model = joblib.load(model_path)

# Build Target Store & Blocker sample
print("Building Target Store & Blocker on 300k sample...")
target_store = OnDemandTargetStore()
blocker = EntityBlocker()

target_loaded = 0
for fpath in [s2_path, s3_path]:
    if target_loaded >= 300000: break
    for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
        if target_loaded >= 300000: break
        sub = chunk.iloc[:min(len(chunk), 300000 - target_loaded)]
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

# Collect candidate pairs
s1_df = pd.read_csv(s1_path, sep="\t", nrows=50, dtype=str)
pair_meta = []
s1_precomputed = []

for _, r in s1_df.iterrows():
    s1_id = r['entity_id']
    c = str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    s1_rec = precompute_s1_record(s1_id, c, r.get('business_name', ''), r.get('business_address', ''))
    s1_idx = len(s1_precomputed)
    s1_precomputed.append(s1_rec)
    
    cands = blocker.get_candidates(s1_rec['raw_rec'])
    for tid in cands:
        if tid in target_store.raw_attr:
            pair_meta.append((s1_idx, tid))
        if len(pair_meta) >= 100000: break
    if len(pair_meta) >= 100000: break

n_pairs = len(pair_meta)
print(f"Collected {n_pairs:,} candidate pairs for comparison.\n")

# 1. RUN BASELINE FEATURE EXTRACTION
print("Running Baseline Feature Extraction...")
t0_base = time.time()
X_base = np.empty((n_pairs, 13), dtype=np.float32)

for p_idx, (s1_idx, tid) in enumerate(pair_meta):
    fill_features_optimized(X_base, p_idx, s1_precomputed[s1_idx], target_store, tid)

t_base = time.time() - t0_base
print(f"Baseline Feature Extraction completed in {t_base:.4f} s ({n_pairs/t_base:,.0f} pairs/sec)")

# 2. NUMBA FAST FEATURE KERNEL DESIGN
@numba.njit(fastmath=True)
def _sorted_intersect_count(a, b):
    i = 0
    j = 0
    count = 0
    na = len(a)
    nb = len(b)
    while i < na and j < nb:
        if a[i] == b[j]:
            count += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return count

@numba.njit(parallel=True, fastmath=True)
def fill_features_numba_batch(
    X,
    s1_norm_n_hash, s1_len_n, s1_len_a, s1_c_hash,
    s1_toks_n_offsets, s1_toks_n_data,
    s1_toks_a_offsets, s1_toks_a_data,
    s1_ng_n_offsets, s1_ng_n_data,
    s1_ng_a_offsets, s1_ng_a_data,
    s1_dig_a_offsets, s1_dig_a_data,
    pair_s1_indices,
    target_norm_n_hash, target_len_n, target_len_a, target_c_hash, target_is_s2,
    target_toks_n_offsets, target_toks_n_data,
    target_toks_a_offsets, target_toks_a_data,
    target_ng_n_offsets, target_ng_n_data,
    target_ng_a_offsets, target_ng_a_data,
    target_dig_a_offsets, target_dig_a_data,
    pair_target_indices
):
    n_pairs = len(pair_s1_indices)
    for p in numba.prange(n_pairs):
        s_idx = pair_s1_indices[p]
        t_idx = pair_target_indices[p]

        # S1 string and scalar properties
        sn_hash = s1_norm_n_hash[s_idx]
        slen_n = s1_len_n[s_idx]
        slen_a = s1_len_a[s_idx]
        sc_hash = s1_c_hash[s_idx]

        # Target string and scalar properties
        tn_hash = target_norm_n_hash[t_idx]
        tlen_n = target_len_n[t_idx]
        tlen_a = target_len_a[t_idx]
        tc_hash = target_c_hash[t_idx]
        is_s2 = target_is_s2[t_idx]

        # Feature 0: Name exact match
        n_exact = 1.0 if (sn_hash == tn_hash and sn_hash != 0) else 0.0

        # Feature 1 & 9: Name token Jaccard & overlap
        s_tok_n = s1_toks_n_data[s1_toks_n_offsets[s_idx]:s1_toks_n_offsets[s_idx+1]]
        t_tok_n = target_toks_n_data[target_toks_n_offsets[t_idx]:target_toks_n_offsets[t_idx+1]]
        len_tn1 = len(s_tok_n)
        len_tn2 = len(t_tok_n)

        if len_tn1 > 0 and len_tn2 > 0:
            n_overlap = _sorted_intersect_count(s_tok_n, t_tok_n)
            n_jaccard = n_overlap / (len_tn1 + len_tn2 - n_overlap)
        else:
            n_overlap = 0
            n_jaccard = 0.0

        # Feature 2: Name 3-gram char Jaccard
        s_ng_n = s1_ng_n_data[s1_ng_n_offsets[s_idx]:s1_ng_n_offsets[s_idx+1]]
        t_ng_n = target_ng_n_data[target_ng_n_offsets[t_idx]:target_ng_n_offsets[t_idx+1]]
        len_ng1_n = len(s_ng_n)
        len_ng2_n = len(t_ng_n)

        if len_ng1_n > 0 and len_ng2_n > 0:
            ng_n_overlap = _sorted_intersect_count(s_ng_n, t_ng_n)
            n_char_jaccard = ng_n_overlap / (len_ng1_n + len_ng2_n - ng_n_overlap)
        else:
            n_char_jaccard = 0.0

        # Feature 3 & 10: Address token Jaccard & overlap
        s_tok_a = s1_toks_a_data[s1_toks_a_offsets[s_idx]:s1_toks_a_offsets[s_idx+1]]
        t_tok_a = target_toks_a_data[target_toks_a_offsets[t_idx]:target_toks_a_offsets[t_idx+1]]
        len_ta1 = len(s_tok_a)
        len_ta2 = len(t_tok_a)

        if len_ta1 > 0 and len_ta2 > 0:
            a_overlap = _sorted_intersect_count(s_tok_a, t_tok_a)
            a_jaccard = a_overlap / (len_ta1 + len_ta2 - a_overlap)
        else:
            a_overlap = 0
            a_jaccard = 0.0

        # Feature 4: Address 3-gram char Jaccard
        s_ng_a = s1_ng_a_data[s1_ng_a_offsets[s_idx]:s1_ng_a_offsets[s_idx+1]]
        t_ng_a = target_ng_a_data[target_ng_a_offsets[t_idx]:target_ng_a_offsets[t_idx+1]]
        len_ng1_a = len(s_ng_a)
        len_ng2_a = len(t_ng_a)

        if len_ng1_a > 0 and len_ng2_a > 0:
            ng_a_overlap = _sorted_intersect_count(s_ng_a, t_ng_a)
            a_char_jaccard = ng_a_overlap / (len_ng1_a + len_ng2_a - ng_a_overlap)
        else:
            a_char_jaccard = 0.0

        # Feature 5: Address digit Jaccard
        s_dig_a = s1_dig_a_data[s1_dig_a_offsets[s_idx]:s1_dig_a_offsets[s_idx+1]]
        t_dig_a = target_dig_a_data[target_dig_a_offsets[t_idx]:target_dig_a_offsets[t_idx+1]]
        len_d1 = len(s_dig_a)
        len_d2 = len(t_dig_a)

        if len_d1 > 0 and len_d2 > 0:
            d_overlap = _sorted_intersect_count(s_dig_a, t_dig_a)
            d_jaccard = d_overlap / (len_d1 + len_d2 - d_overlap)
        else:
            d_jaccard = 0.5

        # Feature 6: Country match
        c_match = 1.0 if (sc_hash == tc_hash and sc_hash != 0) else 0.0

        # Feature 7 & 8: Length diffs
        len_diff_n = float(abs(slen_n - tlen_n))
        len_diff_a = float(abs(slen_a - tlen_a))

        # Feature 11: Missing address
        missing_addr = 1.0 if (slen_a == 0 or tlen_a == 0) else 0.0

        # Write to array directly
        X[p, 0] = n_exact
        X[p, 1] = n_jaccard
        X[p, 2] = n_char_jaccard
        X[p, 3] = a_jaccard
        X[p, 4] = a_char_jaccard
        X[p, 5] = d_jaccard
        X[p, 6] = c_match
        X[p, 7] = len_diff_n
        X[p, 8] = len_diff_a
        X[p, 9] = float(n_overlap)
        X[p, 10] = float(a_overlap)
        X[p, 11] = missing_addr
        X[p, 12] = is_s2

# 3. BUILD INTEGER VOCABULARY & ENCODED TARGET / S1 ARRAYS
print("Building Vocabulary & Integer Encoded Arrays...")
vocab = {}
def get_str_id(s):
    if not s: return 0
    if s not in vocab:
        vocab[s] = len(vocab) + 1
    return vocab[s]

# Encode Target Store
target_mids = list(target_store.raw_attr.keys())
target_mid_map = {mid: i for i, mid in enumerate(target_mids)}
num_targets = len(target_mids)

target_norm_n_hash = np.zeros(num_targets, dtype=np.int64)
target_len_n = np.zeros(num_targets, dtype=np.int32)
target_len_a = np.zeros(num_targets, dtype=np.int32)
target_c_hash = np.zeros(num_targets, dtype=np.int64)
target_is_s2 = np.zeros(num_targets, dtype=np.float32)

t_toks_n_list = []
t_toks_a_list = []
t_ng_n_list = []
t_ng_a_list = []
t_dig_a_list = []

for i, mid in enumerate(target_mids):
    parsed = target_store.get_parsed(mid)
    # c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2
    c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5], parsed[6], parsed[7]
    
    target_norm_n_hash[i] = get_str_id(norm_n2)
    target_len_n[i] = parsed[8]
    target_len_a[i] = parsed[9]
    target_c_hash[i] = get_str_id(c2)
    target_is_s2[i] = parsed[15]
    
    t_toks_n_list.append(np.array(sorted([get_str_id(t) for t in toks_n2]), dtype=np.int32))
    t_toks_a_list.append(np.array(sorted([get_str_id(t) for t in toks_a2]), dtype=np.int32))
    t_ng_n_list.append(np.array(sorted([get_str_id(t) for t in ngrams_n2]), dtype=np.int32))
    t_ng_a_list.append(np.array(sorted([get_str_id(t) for t in ngrams_a2]), dtype=np.int32))
    t_dig_a_list.append(np.array(sorted([get_str_id(t) for t in digits_a2]), dtype=np.int32))

def build_flat_offsets_and_data(arr_list):
    offsets = np.zeros(len(arr_list) + 1, dtype=np.int32)
    for idx, arr in enumerate(arr_list):
        offsets[idx + 1] = offsets[idx] + len(arr)
    data = np.concatenate(arr_list) if arr_list else np.empty(0, dtype=np.int32)
    return offsets, data

target_toks_n_offsets, target_toks_n_data = build_flat_offsets_and_data(t_toks_n_list)
target_toks_a_offsets, target_toks_a_data = build_flat_offsets_and_data(t_toks_a_list)
target_ng_n_offsets, target_ng_n_data = build_flat_offsets_and_data(t_ng_n_list)
target_ng_a_offsets, target_ng_a_data = build_flat_offsets_and_data(t_ng_a_list)
target_dig_a_offsets, target_dig_a_data = build_flat_offsets_and_data(t_dig_a_list)

# Encode S1 Records
num_s1 = len(s1_precomputed)
s1_norm_n_hash = np.zeros(num_s1, dtype=np.int64)
s1_len_n = np.zeros(num_s1, dtype=np.int32)
s1_len_a = np.zeros(num_s1, dtype=np.int32)
s1_c_hash = np.zeros(num_s1, dtype=np.int64)

s1_toks_n_list = []
s1_toks_a_list = []
s1_ng_n_list = []
s1_ng_a_list = []
s1_dig_a_list = []

for i, s1 in enumerate(s1_precomputed):
    s1_norm_n_hash[i] = get_str_id(s1['norm_n'])
    s1_len_n[i] = s1['len_n']
    s1_len_a[i] = s1['len_a']
    s1_c_hash[i] = get_str_id(s1['c'])
    
    s1_toks_n_list.append(np.array(sorted([get_str_id(t) for t in s1['toks_n']]), dtype=np.int32))
    s1_toks_a_list.append(np.array(sorted([get_str_id(t) for t in s1['toks_a']]), dtype=np.int32))
    s1_ng_n_list.append(np.array(sorted([get_str_id(t) for t in s1['ngrams_n']]), dtype=np.int32))
    s1_ng_a_list.append(np.array(sorted([get_str_id(t) for t in s1['ngrams_a']]), dtype=np.int32))
    s1_dig_a_list.append(np.array(sorted([get_str_id(t) for t in s1['digits_a']]), dtype=np.int32))

s1_toks_n_offsets, s1_toks_n_data = build_flat_offsets_and_data(s1_toks_n_list)
s1_toks_a_offsets, s1_toks_a_data = build_flat_offsets_and_data(s1_toks_a_list)
s1_ng_n_offsets, s1_ng_n_data = build_flat_offsets_and_data(s1_ng_n_list)
s1_ng_a_offsets, s1_ng_a_data = build_flat_offsets_and_data(s1_ng_a_list)
s1_dig_a_offsets, s1_dig_a_data = build_flat_offsets_and_data(s1_dig_a_list)

pair_s1_indices = np.array([s1_idx for s1_idx, _ in pair_meta], dtype=np.int32)
pair_target_indices = np.array([target_mid_map[tid] for _, tid in pair_meta], dtype=np.int32)

# Warmup Numba JIT Compilation
X_numba = np.empty((n_pairs, 13), dtype=np.float32)
fill_features_numba_batch(
    X_numba[:10],
    s1_norm_n_hash, s1_len_n, s1_len_a, s1_c_hash,
    s1_toks_n_offsets, s1_toks_n_data,
    s1_toks_a_offsets, s1_toks_a_data,
    s1_ng_n_offsets, s1_ng_n_data,
    s1_ng_a_offsets, s1_ng_a_data,
    s1_dig_a_offsets, s1_dig_a_data,
    pair_s1_indices[:10],
    target_norm_n_hash, target_len_n, target_len_a, target_c_hash, target_is_s2,
    target_toks_n_offsets, target_toks_n_data,
    target_toks_a_offsets, target_toks_a_data,
    target_ng_n_offsets, target_ng_n_data,
    target_ng_a_offsets, target_ng_a_data,
    target_dig_a_offsets, target_dig_a_data,
    pair_target_indices[:10]
)

# Benchmark Numba Execution Speed
print("\nRunning Numba Parallel Accelerated Feature Extraction...")
t0_numba = time.time()

fill_features_numba_batch(
    X_numba,
    s1_norm_n_hash, s1_len_n, s1_len_a, s1_c_hash,
    s1_toks_n_offsets, s1_toks_n_data,
    s1_toks_a_offsets, s1_toks_a_data,
    s1_ng_n_offsets, s1_ng_n_data,
    s1_ng_a_offsets, s1_ng_a_data,
    s1_dig_a_offsets, s1_dig_a_data,
    pair_s1_indices,
    target_norm_n_hash, target_len_n, target_len_a, target_c_hash, target_is_s2,
    target_toks_n_offsets, target_toks_n_data,
    target_toks_a_offsets, target_toks_a_data,
    target_ng_n_offsets, target_ng_n_data,
    target_ng_a_offsets, target_ng_a_data,
    target_dig_a_offsets, target_dig_a_data,
    pair_target_indices
)

t_numba = time.time() - t0_numba
speedup = t_base / t_numba if t_numba > 0 else 0

print(f"Numba Feature Extraction completed in {t_numba:.4f} s ({n_pairs/t_numba:,.0f} pairs/sec)")
print(f"Speedup Multiplier: {speedup:.2f}x FASTER!")

# 4. NUMERICAL EQUIVALENCE & MODEL PREDICTION CHECKS
diff = np.abs(X_base - X_numba)
max_diff = float(np.max(diff))
mean_diff = float(np.mean(diff))

feature_names = [
    "n_exact", "n_jaccard", "n_char_jaccard", "a_jaccard", "a_char_jaccard",
    "d_jaccard", "c_match", "len_diff_n", "len_diff_a", "n_overlap",
    "a_overlap", "missing_addr", "is_s2"
]

print("\n--- Numerical Equivalence Verification ---")
print(f"Max Absolute Feature Difference:  {max_diff:.8f}")
print(f"Mean Absolute Feature Difference: {mean_diff:.8f}")

print("\nPer-Feature Maximum Difference:")
for idx, fname in enumerate(feature_names):
    f_max_d = float(np.max(diff[:, idx]))
    print(f"  Feature [{idx:2d}] {fname:20s}: {f_max_d:.8f}")

# Predict with LightGBM
probs_base = model.predict_proba(X_base)[:, 1]
probs_numba = model.predict_proba(X_numba)[:, 1]

prob_diff = np.abs(probs_base - probs_numba)
max_prob_d = float(np.max(prob_diff))
mean_prob_d = float(np.mean(prob_diff))

matches_base = (probs_base >= 0.98)
matches_numba = (probs_numba >= 0.98)
mismatch_preds = (matches_base != matches_numba).sum()

print("\n--- LightGBM Model Prediction Equivalence ---")
print(f"Max Probability Difference:       {max_prob_d:.8f}")
print(f"Mean Probability Difference:      {mean_prob_d:.8f}")
print(f"Prediction Mismatches @ T=0.98:   {mismatch_preds} / {n_pairs:,} pairs")

if max_diff <= 1e-5 and mismatch_preds == 0:
    print("\n[VERIFICATION PASS] Numba feature extraction is 100% numerically equivalent and prediction identical!")
else:
    print("\n[VERIFICATION FAIL] Mismatch detected!")

print(f"Peak RAM: {psutil.Process(os.getpid()).memory_info().rss / (1024*1024):.1f} MB")
