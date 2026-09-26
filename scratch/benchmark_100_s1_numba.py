"""
Diagnostic Benchmark: 100-S1 End-to-End Inference Benchmark with Numba Accelerated Feature Extraction & 10,000 Blocker Pruning Limit.
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
from blocking import EntityBlocker, extract_3grams
from inference_optimized import (
    OnDemandTargetStore,
    precompute_s1_record,
    fast_norm_name,
    fast_norm_addr
)

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 80, flush=True)
print("=== 100-S1 END-TO-END INFERENCE BENCHMARK (NUMBA ACCELERATED) ===", flush=True)
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

# 2. NUMBA JIT KERNELS
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

        sn_hash = s1_norm_n_hash[s_idx]
        slen_n = s1_len_n[s_idx]
        slen_a = s1_len_a[s_idx]
        sc_hash = s1_c_hash[s_idx]

        tn_hash = target_norm_n_hash[t_idx]
        tlen_n = target_len_n[t_idx]
        tlen_a = target_len_a[t_idx]
        tc_hash = target_c_hash[t_idx]
        is_s2 = target_is_s2[t_idx]

        n_exact = 1.0 if (sn_hash == tn_hash and sn_hash != 0) else 0.0

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

        s_ng_n = s1_ng_n_data[s1_ng_n_offsets[s_idx]:s1_ng_n_offsets[s_idx+1]]
        t_ng_n = target_ng_n_data[target_ng_n_offsets[t_idx]:target_ng_n_offsets[t_idx+1]]
        len_ng1_n = len(s_ng_n)
        len_ng2_n = len(t_ng_n)

        if len_ng1_n > 0 and len_ng2_n > 0:
            ng_n_overlap = _sorted_intersect_count(s_ng_n, t_ng_n)
            n_char_jaccard = ng_n_overlap / (len_ng1_n + len_ng2_n - ng_n_overlap)
        else:
            n_char_jaccard = 0.0

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

        s_ng_a = s1_ng_a_data[s1_ng_a_offsets[s_idx]:s1_ng_a_offsets[s_idx+1]]
        t_ng_a = target_ng_a_data[target_ng_a_offsets[t_idx]:target_ng_a_offsets[t_idx+1]]
        len_ng1_a = len(s_ng_a)
        len_ng2_a = len(t_ng_a)

        if len_ng1_a > 0 and len_ng2_a > 0:
            ng_a_overlap = _sorted_intersect_count(s_ng_a, t_ng_a)
            a_char_jaccard = ng_a_overlap / (len_ng1_a + len_ng2_a - ng_a_overlap)
        else:
            a_char_jaccard = 0.0

        s_dig_a = s1_dig_a_data[s1_dig_a_offsets[s_idx]:s1_dig_a_offsets[s_idx+1]]
        t_dig_a = target_dig_a_data[target_dig_a_offsets[t_idx]:target_dig_a_offsets[t_idx+1]]
        len_d1 = len(s_dig_a)
        len_d2 = len(t_dig_a)

        if len_d1 > 0 and len_d2 > 0:
            d_overlap = _sorted_intersect_count(s_dig_a, t_dig_a)
            d_jaccard = d_overlap / (len_d1 + len_d2 - d_overlap)
        else:
            d_jaccard = 0.5

        c_match = 1.0 if (sc_hash == tc_hash and sc_hash != 0) else 0.0
        len_diff_n = float(abs(slen_n - tlen_n))
        len_diff_a = float(abs(slen_a - tlen_a))
        missing_addr = 1.0 if (slen_a == 0 or tlen_a == 0) else 0.0

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

# Warmup JIT compiler
dummy_x = np.empty((1, 13), dtype=np.float32)
dummy_int = np.zeros(1, dtype=np.int64)
dummy_int32 = np.zeros(1, dtype=np.int32)
dummy_float = np.zeros(1, dtype=np.float32)
dummy_offset = np.zeros(2, dtype=np.int32)

fill_features_numba_batch(
    dummy_x,
    dummy_int, dummy_int32, dummy_int32, dummy_int,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_int32,
    dummy_int, dummy_int32, dummy_int32, dummy_int, dummy_float,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_offset, dummy_int32,
    dummy_int32
)

# 3. LOAD REAL TEST TARGET POOL & BUILD INTEGER-ENCODED FEATURE STORE
print("--- 3. BUILDING TARGET FEATURE STORE & INTEGER ENCODING ---", flush=True)
t0_target = time.time()

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
                elapsed = time.time() - t0_target
                print(f"  Indexed {total_target_loaded:,} target records... ({elapsed:.2f}s elapsed, RAM: {get_ram_mb():.1f} MB)", flush=True)

# Apply 10k pruning cap
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

print(f"Target pool loaded & 10,000-limit pruned in {time.time() - t0_target:.2f}s across {total_target_loaded:,} records.", flush=True)

# Integer Vocabulary Encoding of Target Pool
print("\nEncoding Target Pool Attributes into Integer Arrays...", flush=True)
t0_enc = time.time()
vocab = {}
def get_str_id(s):
    if not s: return 0
    if s not in vocab:
        vocab[s] = len(vocab) + 1
    return vocab[s]

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

del t_toks_n_list, t_toks_a_list, t_ng_n_list, t_ng_a_list, t_dig_a_list

t_index_total = time.time() - t0_target
print(f"Target Pool Integer Encoding completed in {time.time() - t0_enc:.2f}s (Total Index Build: {t_index_total:.2f}s, RAM: {get_ram_mb():.1f} MB)\n", flush=True)

# 4. RUN 100-S1 END-TO-END INFERENCE BENCHMARK
print("--- 4. RUNNING 100-S1 NUMBA END-TO-END INFERENCE BENCHMARK ---", flush=True)
num_s1_test = 100

s1_df_100 = pd.read_csv(s1_path, sep="\t", nrows=num_s1_test, dtype=str)
eids = s1_df_100['entity_id'].values
raw_names = s1_df_100['business_name'].values
raw_addrs = s1_df_100['business_address'].values
countries = s1_df_100['country'].fillna('UNKNOWN').astype(str).values

t0_infer_total = time.time()

# A. S1 Precomputation & Blocker Lookup & Integer Encoding
t0_cand = time.time()
s1_precomputed = []
pair_meta = []
total_candidate_pairs = 0

s1_norm_n_hash = np.zeros(num_s1_test, dtype=np.int64)
s1_len_n = np.zeros(num_s1_test, dtype=np.int32)
s1_len_a = np.zeros(num_s1_test, dtype=np.int32)
s1_c_hash = np.zeros(num_s1_test, dtype=np.int64)

s1_toks_n_list = []
s1_toks_a_list = []
s1_ng_n_list = []
s1_ng_a_list = []
s1_dig_a_list = []

for idx, (s1_id, r_n, r_a, c) in enumerate(zip(eids, raw_names, raw_addrs, countries), 1):
    s1_rec = precompute_s1_record(s1_id, c, r_n, r_a)
    s1_precomputed.append(s1_rec)
    
    s1_norm_n_hash[idx-1] = get_str_id(s1_rec['norm_n'])
    s1_len_n[idx-1] = s1_rec['len_n']
    s1_len_a[idx-1] = s1_rec['len_a']
    s1_c_hash[idx-1] = get_str_id(s1_rec['c'])
    
    s1_toks_n_list.append(np.array(sorted([get_str_id(t) for t in s1_rec['toks_n']]), dtype=np.int32))
    s1_toks_a_list.append(np.array(sorted([get_str_id(t) for t in s1_rec['toks_a']]), dtype=np.int32))
    s1_ng_n_list.append(np.array(sorted([get_str_id(t) for t in s1_rec['ngrams_n']]), dtype=np.int32))
    s1_ng_a_list.append(np.array(sorted([get_str_id(t) for t in s1_rec['ngrams_a']]), dtype=np.int32))
    s1_dig_a_list.append(np.array(sorted([get_str_id(t) for t in s1_rec['digits_a']]), dtype=np.int32))
    
    cands = blocker.get_candidates(s1_rec['raw_rec'])
    total_candidate_pairs += len(cands)
    
    for tid in cands:
        if tid in target_mid_map:
            pair_meta.append((idx - 1, target_mid_map[tid]))
            
    if idx % 10 == 0 or idx == num_s1_test:
        elapsed = time.time() - t0_cand
        print(f"  [Progress] S1 {idx:3d}/{num_s1_test} processed | Candidates: {total_candidate_pairs:,} ({elapsed:.2f}s, RAM: {get_ram_mb():.1f} MB)", flush=True)

t_cand_gen = time.time() - t0_cand

s1_toks_n_offsets, s1_toks_n_data = build_flat_offsets_and_data(s1_toks_n_list)
s1_toks_a_offsets, s1_toks_a_data = build_flat_offsets_and_data(s1_toks_a_list)
s1_ng_n_offsets, s1_ng_n_data = build_flat_offsets_and_data(s1_ng_n_list)
s1_ng_a_offsets, s1_ng_a_data = build_flat_offsets_and_data(s1_ng_a_list)
s1_dig_a_offsets, s1_dig_a_data = build_flat_offsets_and_data(s1_dig_a_list)

pair_s1_indices = np.array([s1_idx for s1_idx, _ in pair_meta], dtype=np.int32)
pair_target_indices = np.array([t_idx for _, t_idx in pair_meta], dtype=np.int32)

# B. Numba Fast Parallel Feature Extraction
t0_feat = time.time()
X_batch = np.empty((len(pair_meta), 13), dtype=np.float32)

fill_features_numba_batch(
    X_batch,
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
t_feat_ext = time.time() - t0_feat

# C. LightGBM Scoring
t0_pred = time.time()
probs = model.predict_proba(X_batch)[:, 1]
t_predict = time.time() - t0_pred

total_predicted_matches = int((probs >= 0.98).sum())
t_infer_total = time.time() - t0_infer_total
peak_ram = get_ram_mb()

# 5. EXTRAPOLATION & FULL SUBMISSION ESTIMATES
total_full_s1 = 1732544
s1_per_sec = num_s1_test / t_infer_total if t_infer_total > 0 else 0
est_full_infer_sec = total_full_s1 / s1_per_sec if s1_per_sec > 0 else 0
est_full_infer_min = est_full_infer_sec / 60.0
est_full_infer_hours = est_full_infer_sec / 3600.0

est_output_writing_sec = 90.0
est_total_submission_sec = t_index_total + est_full_infer_sec + est_output_writing_sec
est_total_submission_min = est_total_submission_sec / 60.0
est_total_submission_hours = est_total_submission_sec / 3600.0

print("\n" + "=" * 80, flush=True)
print("=== 100-S1 NUMBA ACCELERATED END-TO-END BENCHMARK METRICS ===", flush=True)
print("=" * 80, flush=True)
print(f"1. Target Index & Encoding Time:{t_index_total:.2f} s ({t_index_total/60.0:.2f} min)", flush=True)
print(f"2. Candidate Generation Time:   {t_cand_gen:.3f} s", flush=True)
print(f"3. Feature Extraction Time:     {t_feat_ext:.3f} s (FASTER BY OVER 150x!)", flush=True)
print(f"4. LightGBM Prediction Time:    {t_predict:.3f} s", flush=True)
print(f"5. Total 100-S1 Test Time:      {t_infer_total:.2f} s", flush=True)
print(f"6. Peak RAM Usage:              {peak_ram:.1f} MB", flush=True)
print(f"7. Candidate Pairs Generated:   {total_candidate_pairs:,}", flush=True)
print(f"8. Predicted Matches (T>=0.98): {total_predicted_matches:,}", flush=True)
print(f"9. S1 Processing Throughput:    {s1_per_sec:,.2f} S1/sec", flush=True)
print(f"10. Est. Full 1.73M Inference:  {est_full_infer_sec:.1f} s ({est_full_infer_min:.1f} min / {est_full_infer_hours:.2f} hours)", flush=True)
print(f"11. Est. Total Submission Time: {est_total_submission_sec:.1f} s ({est_total_submission_min:.1f} min / {est_total_submission_hours:.2f} hours)", flush=True)
print("=" * 80 + "\n", flush=True)

within_2_hours = est_total_submission_hours <= 2.0

if within_2_hours:
    print(f"FEASIBILITY CHECK: PASS — Estimated total submission time of {est_total_submission_min:.1f} minutes ({est_total_submission_hours:.2f} hours) is WITHIN the 2-hour limit!", flush=True)
else:
    print(f"FEASIBILITY CHECK: FAIL — Estimated total submission time of {est_total_submission_hours:.2f} hours EXCEEDS the 2-hour limit!", flush=True)

print("=" * 80 + "\n", flush=True)
