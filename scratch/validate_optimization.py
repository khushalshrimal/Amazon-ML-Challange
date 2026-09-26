"""
Validation Comparison Suite: Original vs Optimized Inference Pipeline.
Runs on a deterministic sample of 1,000 S1 entities to verify 100% numerical and candidate equivalence.
"""

import os
import sys
import time
import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

base_dir = Path(r"c:\Users\khush\OneDrive\Desktop\amzon ml challenge")
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
sys.path.append(str(src_dir))

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

print("=== VALIDATION SUITE: ORIGINAL VS OPTIMIZED INFERENCE ===", flush=True)

# 1. LOAD MODEL ARTIFACT
model_path = base_dir / "models" / "lightgbm_model.joblib"
print(f"Loading LightGBM model from {model_path}...", flush=True)
model = joblib.load(model_path)

# 2. LOAD 1,000 S1 ENTITIES & GROUND TRUTH
train_dir = base_dir / "dataset" / "train"
gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", nrows=1000, dtype=str)

val_gt_map = {}
val_gt_pairs = set()
s1_needed = set()
s2_needed = set()
s3_needed = set()

for _, r in gt_df.iterrows():
    s1_id = r['source1_entity_id']
    s1_needed.add(s1_id)
    m_str = str(r['matched_entity_ids']).strip() if pd.notna(r['matched_entity_ids']) else ""
    m_set = set(m.strip() for m in m_str.split(',') if m.strip()) if m_str else set()
    val_gt_map[s1_id] = m_set
    for m in m_set:
        val_gt_pairs.add((s1_id, m))
        if m.startswith("S2-"):
            s2_needed.add(m)
        elif m.startswith("S3-"):
            s3_needed.add(m)

print(f"Loaded 1,000 GT S1 entities, {len(val_gt_pairs):,} true ground-truth pairs.", flush=True)

# Load S1 records
s1_records = []
for chunk in pd.read_csv(train_dir / "train_source1.tsv", sep="\t", chunksize=100000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s1_needed)]
    s1_records.append(m)
    if len(pd.concat(s1_records)) >= len(s1_needed):
        break

s1_val_df = pd.concat(s1_records, ignore_index=True)

# Load Target Pool
s2_bg = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=100000, dtype=str)
s3_bg = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=100000, dtype=str)

target_df = pd.concat([s2_bg, s3_bg], ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"Target pool contains {len(target_df):,} unique target records.", flush=True)

# 3. BUILD BLOCKER INDEX & TARGET FEATURE STORE
print("Building Blocker Index & Target Feature Store...", flush=True)
blocker = EntityBlocker()

target_attr_orig = {}
target_store_opt = OnDemandTargetStore()

for r in target_df.to_dict('records'):
    mid = r['entity_id']
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    c = str(r.get('country', '')) if pd.notna(r.get('country')) else "UNKNOWN"
    norm_n = fast_norm_name(raw_n)
    norm_a = fast_norm_addr(raw_a)
    
    # Baseline attribute format
    target_attr_orig[mid] = f"{c}\t{norm_n}\t{norm_a}"
    
    # Optimized Feature Store
    target_store_opt.add_target_raw(mid, c, norm_n, norm_a)
    
    # Add to Blocker
    blocker.add_target_record(mid, c, norm_n, norm_a)

# Original feature extraction helper
def extract_fast_features_orig(a1, mid2):
    rec_str = target_attr_orig[mid2]
    c2, norm_n2, norm_a2 = rec_str.split('\t')
    
    toks_n2 = set(norm_n2.split())
    toks_a2 = set(norm_a2.split())
    ngrams_n2 = extract_ngrams(norm_n2, 3)
    ngrams_a2 = extract_ngrams(norm_a2, 3)
    digits_a2 = extract_digits(norm_a2)
    
    n_exact = 1.0 if a1['norm_n'] == norm_n2 and a1['norm_n'] != "" else 0.0
    
    n_overlap = len(a1['toks_n'] & toks_n2)
    a_overlap = len(a1['toks_a'] & toks_a2)
    
    n_jaccard = n_overlap / len(a1['toks_n'] | toks_n2) if a1['toks_n'] and toks_n2 else 0.0
    a_jaccard = a_overlap / len(a1['toks_a'] | toks_a2) if a1['toks_a'] and toks_a2 else 0.0

    ng_n1, ng_n2 = a1['ngrams_n'], ngrams_n2
    n_char_jaccard = len(ng_n1 & ng_n2) / len(ng_n1 | ng_n2) if ng_n1 and ng_n2 else 0.0

    ng_a1, ng_a2 = a1['ngrams_a'], ngrams_a2
    a_char_jaccard = len(ng_a1 & ng_a2) / len(ng_a1 | ng_a2) if ng_a1 and ng_a2 else 0.0

    d1, d2 = a1['digits_a'], digits_a2
    d_jaccard = len(d1 & d2) / len(d1 | d2) if d1 and d2 else 0.5

    c_match = 1.0 if a1['c'] == c2 and a1['c'] != "" else 0.0
    len_diff_n = float(abs(len(a1['norm_n']) - len(norm_n2)))
    len_diff_a = float(abs(len(a1['norm_a']) - len(norm_a2)))
    missing_addr = 1.0 if not a1['norm_a'] or not norm_a2 else 0.0
    is_s2 = 1.0 if mid2.startswith('S2-') else 0.0

    return [
        n_exact, n_jaccard, n_char_jaccard, a_jaccard, a_char_jaccard,
        d_jaccard, c_match, len_diff_n, len_diff_a, float(n_overlap),
        float(a_overlap), missing_addr, is_s2
    ]

# 4. RUN BOTH PIPELINES & COMPARE
print("\n--- Running Validation Comparison ---", flush=True)

s1_records_dict = s1_val_df.to_dict('records')

cands_orig_all = {}
cands_opt_all = {}

X_orig_list = []
X_opt_list = []
pair_meta_list = []

t0_orig = time.time()
for r in s1_records_dict:
    s1_id = r['entity_id']
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    c = str(r.get('country', '')) if pd.notna(r.get('country')) else "UNKNOWN"
    norm_n = fast_norm_name(raw_n)
    norm_a = fast_norm_addr(raw_a)
    
    a1_orig = {
        'norm_n': norm_n,
        'toks_n': set(norm_n.split()),
        'ngrams_n': extract_ngrams(norm_n, 3),
        'norm_a': norm_a,
        'toks_a': set(norm_a.split()),
        'ngrams_a': extract_ngrams(norm_a, 3),
        'digits_a': extract_digits(norm_a),
        'c': c
    }
    raw_rec = {'entity_id': s1_id, 'business_name': norm_n, 'business_address': norm_a, 'country': c}
    cands = blocker.get_candidates(raw_rec)
    cands_orig_all[s1_id] = cands
    
    for tid in cands:
        if tid in target_attr_orig:
            fvec = extract_fast_features_orig(a1_orig, tid)
            X_orig_list.append(fvec)
            pair_meta_list.append((s1_id, tid))

t_orig_elapsed = time.time() - t0_orig

t0_opt = time.time()
pair_idx = 0
X_opt_arr = np.empty((len(pair_meta_list), 13), dtype=np.float32)

for r in s1_records_dict:
    s1_id = r['entity_id']
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    c = str(r.get('country', '')) if pd.notna(r.get('country')) else "UNKNOWN"
    
    s1_opt = precompute_s1_record(s1_id, c, raw_n, raw_a)
    cands = blocker.get_candidates(s1_opt['raw_rec'])
    cands_opt_all[s1_id] = cands
    
    for tid in cands:
        if tid in target_store_opt.raw_attr:
            fill_features_optimized(X_opt_arr, pair_idx, s1_opt, target_store_opt, tid)
            pair_idx += 1

t_opt_elapsed = time.time() - t0_opt

X_orig_arr = np.array(X_orig_list, dtype=np.float32)

# 5. VERIFICATION CHECKS

# Check 1: Blocker candidate sets
candidate_mismatches = 0
for s1_id in cands_orig_all:
    if cands_orig_all[s1_id] != cands_opt_all[s1_id]:
        candidate_mismatches += 1

print(f"\n1. Candidate Set Mismatches across 1,000 S1 records: {candidate_mismatches}")
if candidate_mismatches > 0:
    print("CRITICAL ERROR: Candidate sets differ! Stopping validation.", flush=True)
    sys.exit(1)
else:
    print("   [PASS] All 1,000 candidate sets are 100% IDENTICAL.", flush=True)

# Check 2: Total candidate counts & Recall
tot_cands_orig = sum(len(c) for c in cands_orig_all.values())
tot_cands_opt = sum(len(c) for c in cands_opt_all.values())

gt_captured_orig = sum(len(cands_orig_all[s1_id] & val_gt_map.get(s1_id, set())) for s1_id in cands_orig_all)
gt_captured_opt = sum(len(cands_opt_all[s1_id] & val_gt_map.get(s1_id, set())) for s1_id in cands_opt_all)

print(f"\n2. Total Candidate Pairs: Original={tot_cands_orig:,} | Optimized={tot_cands_opt:,}")
print(f"   Candidate Recall: Original={gt_captured_orig/len(val_gt_pairs)*100:.2f}% | Optimized={gt_captured_opt/len(val_gt_pairs)*100:.2f}%")
assert tot_cands_orig == tot_cands_opt, "Candidate counts must match!"
print("   [PASS] Candidate counts and recall match 100%.", flush=True)

# Check 3: Numerical Feature Equivalence
diff_matrix = np.abs(X_orig_arr - X_opt_arr)
max_feat_diff = float(np.max(diff_matrix))
mean_feat_diff = float(np.mean(diff_matrix))

feature_names = [
    "n_exact", "n_jaccard", "n_char_jaccard", "a_jaccard", "a_char_jaccard",
    "d_jaccard", "c_match", "len_diff_n", "len_diff_a", "n_overlap",
    "a_overlap", "missing_addr", "is_s2"
]

print(f"\n3. Feature Numerical Equivalence ({len(X_orig_arr):,} pairs x 13 features):")
print(f"   Max Absolute Feature Difference:  {max_feat_diff:.8f}")
print(f"   Mean Absolute Feature Difference: {mean_feat_diff:.8f}")

print("\n   Per-Feature Maximum Absolute Difference:")
for i, name in enumerate(feature_names):
    max_d = float(np.max(diff_matrix[:, i]))
    print(f"     Feature [{i:2d}] {name:20s}: {max_d:.8f}")

if max_feat_diff > 1e-5:
    print(f"CRITICAL ERROR: Max feature difference {max_feat_diff} exceeds tolerance 1e-5!", flush=True)
    sys.exit(1)
else:
    print("   [PASS] Feature values are numerically equivalent within 1e-5 float32 tolerance.", flush=True)

# Check 4: LightGBM Predictions & Threshold 0.98 Matching
probs_orig = model.predict_proba(X_orig_arr)[:, 1]
probs_opt = model.predict_proba(X_opt_arr)[:, 1]

prob_diff = np.abs(probs_orig - probs_opt)
max_prob_diff = float(np.max(prob_diff))
mean_prob_diff = float(np.mean(prob_diff))

print(f"\n4. LightGBM Model Prediction Equivalence:")
print(f"   Max Absolute Probability Difference:  {max_prob_diff:.8f}")
print(f"   Mean Absolute Probability Difference: {mean_prob_diff:.8f}")

preds_orig_map = defaultdict(set)
preds_opt_map = defaultdict(set)

for (s1_id, tid), p_orig, p_opt in zip(pair_meta_list, probs_orig, probs_opt):
    if p_orig >= 0.98:
        preds_orig_map[s1_id].add(tid)
    if p_opt >= 0.98:
        preds_opt_map[s1_id].add(tid)

pred_mismatches = 0
for s1_id in s1_records_dict:
    sid = s1_id['entity_id']
    if preds_orig_map[sid] != preds_opt_map[sid]:
        pred_mismatches += 1

print(f"\n5. Final Predictions at Threshold 0.98:")
print(f"   Prediction Target Mismatches: {pred_mismatches} / 1,000 S1 records")
if pred_mismatches > 0:
    print("CRITICAL ERROR: Predictions at threshold 0.98 differ!", flush=True)
    sys.exit(1)
else:
    print("   [PASS] Predictions at threshold 0.98 are 100% IDENTICAL.", flush=True)

print(f"\n6. Speedup Comparison on 1,000 S1 sample:")
print(f"   Original Runtime:  {t_orig_elapsed:.3f} s ({len(pair_meta_list)/t_orig_elapsed:,.0f} pairs/sec)")
print(f"   Optimized Runtime: {t_opt_elapsed:.3f} s ({len(pair_meta_list)/t_opt_elapsed:,.0f} pairs/sec)")
print(f"   Speedup Multiplier: {t_orig_elapsed / t_opt_elapsed:.2f}x faster!")

print("\n=== VALIDATION SUCCESSFUL! ALL TESTS PASSED! ===", flush=True)
