import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier

base_dir = Path(r"c:\Users\khush\OneDrive\Desktop\amzon ml challenge")
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from blocking import EntityBlocker
from features import extract_pair_features_precomputed
from evaluate import evaluate_predictions

print("=== PHASE 4: FAST TRAINING & THRESHOLD CALIBRATION PIPELINE ===", flush=True)

# 1. Load Ground Truth (20,000 S1 Entities)
train_dir = base_dir / "dataset" / "train"
gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", nrows=20000, dtype=str)

val_gt_map = {}
val_gt_pairs = set()
s1_needed = set()
s2_needed = set()
s3_needed = set()

for _, r in gt_df.iterrows():
    s1_id = r['source1_entity_id']
    s1_needed.add(s1_id)
    m_str = str(r['matched_entity_ids']).strip() if pd.notna(r['matched_entity_ids']) else ""
    if m_str:
        m_set = set(m.strip() for m in m_str.split(',') if m.strip())
    else:
        m_set = set()
    val_gt_map[s1_id] = m_set
    for m in m_set:
        val_gt_pairs.add((s1_id, m))
        if m.startswith("S2-"):
            s2_needed.add(m)
        elif m.startswith("S3-"):
            s3_needed.add(m)

print(f"Loaded {len(val_gt_map):,} GT S1 entities, {len(val_gt_pairs):,} true match pairs.", flush=True)

# 2. Load S1 validation entity records
s1_records = []
for chunk in pd.read_csv(train_dir / "train_source1.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s1_needed)]
    s1_records.append(m)
    if len(pd.concat(s1_records)) >= len(s1_needed):
        break

s1_val_df = pd.concat(s1_records, ignore_index=True)
s1_val_dict = s1_val_df.set_index('entity_id').to_dict(orient='index')

# Precompute S1 attributes
s1_attr = {}
for s1_id, r in s1_val_dict.items():
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    c = str(r.get('country', '')) if pd.notna(r.get('country')) else "UNKNOWN"
    norm_n = normalize_name(raw_n)
    norm_a = normalize_address(raw_a)
    s1_attr[s1_id] = {
        'norm_n': norm_n,
        'toks_n': set(norm_n.split()),
        'norm_a': norm_a,
        'toks_a': set(norm_a.split()),
        'c': c,
        'raw_rec': r
    }

# 3. Fetch S2 and S3 target pool
print("Loading target pool...", flush=True)
s2_records = []
for chunk in pd.read_csv(train_dir / "train_source2.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s2_needed)]
    s2_records.append(m)
    if len(pd.concat(s2_records)) >= len(s2_needed):
        break

s3_records = []
for chunk in pd.read_csv(train_dir / "train_source3.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s3_needed)]
    s3_records.append(m)
    if len(pd.concat(s3_records)) >= len(s3_needed):
        break

s2_bg = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=200000, dtype=str)
s3_bg = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=200000, dtype=str)

target_pool = pd.concat([pd.concat(s2_records), pd.concat(s3_records), s2_bg, s3_bg], ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"Target pool contains {len(target_pool):,} unique S2/S3 records.", flush=True)

# Build Blocker Index
print("Building Blocker Index...", flush=True)
blocker = EntityBlocker()
blocker.build_index(target_pool.to_dict('records'))

# Precompute target pool attributes
print("Precomputing target pool attributes...", flush=True)
target_attr = {}
for r in target_pool.to_dict('records'):
    mid = r['entity_id']
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    c = str(r.get('country', '')) if pd.notna(r.get('country')) else "UNKNOWN"
    norm_n = normalize_name(raw_n)
    norm_a = normalize_address(raw_a)
    target_attr[mid] = {
        'norm_n': norm_n,
        'toks_n': set(norm_n.split()),
        'norm_a': norm_a,
        'toks_a': set(norm_a.split()),
        'c': c
    }

# 4. Generate Candidates and Build Feature Dataset
print("Generating candidate pairs and extracting features with fast skips...", flush=True)
t0 = time.time()

# Split S1 entities into Train (14,000) and Val (6,000)
all_s1_ids = list(s1_attr.keys())
np.random.seed(42)
np.random.shuffle(all_s1_ids)

train_s1_ids = set(all_s1_ids[:14000])
val_s1_ids = set(all_s1_ids[14000:])

X_train = []
y_train = []

val_candidates = {}  # s1_id -> list of (target_id, feature_dict)

train_pos_count = 0
train_neg_count = 0

processed_count = 0

for s1_id, a1 in s1_attr.items():
    processed_count += 1
    if processed_count % 5000 == 0:
        print(f"Processed {processed_count:,} / {len(s1_attr):,} S1 entities ({time.time()-t0:.1f}s)...", flush=True)

    is_train = s1_id in train_s1_ids
    gt_set = val_gt_map.get(s1_id, set())
    
    cands = blocker.get_candidates(a1['raw_rec'])
    
    # Always include all true matches in candidate set
    full_cands = cands | gt_set

    cands_with_feats = []
    train_neg_candidates = []
    
    for tid in full_cands:
        if tid not in target_attr:
            continue
        a2 = target_attr[tid]
        is_match = 1 if tid in gt_set else 0

        feat = extract_pair_features_precomputed(
            a1['norm_n'], a1['toks_n'], a1['norm_a'], a1['toks_a'], a1['c'],
            a2['norm_n'], a2['toks_n'], a2['norm_a'], a2['toks_a'], a2['c'],
            tid,
            skip_slow_sim=True
        )

        if is_train:
            if is_match == 1:
                X_train.append(list(feat.values()))
                y_train.append(1)
                train_pos_count += 1
            else:
                # Keep negatives if name/addr jaccard > 0 or prefix match
                if feat['name_jaccard'] > 0 or feat['addr_jaccard'] > 0 or feat['name_prefix3_match'] == 1.0:
                    train_neg_candidates.append(feat)
        else:
            # For validation: include non-zero similarity or true match candidates
            if is_match == 1 or feat['name_jaccard'] > 0 or feat['addr_jaccard'] > 0 or feat['name_prefix3_match'] == 1.0:
                cands_with_feats.append((tid, list(feat.values())))

    if is_train and train_neg_candidates:
        max_negs = max(10, len(gt_set) * 10)
        if len(train_neg_candidates) > max_negs:
            # pick random subset
            idx_sample = np.random.choice(len(train_neg_candidates), size=max_negs, replace=False)
            sampled_negs = [train_neg_candidates[i] for i in idx_sample]
        else:
            sampled_negs = train_neg_candidates
        for f in sampled_negs:
            X_train.append(list(f.values()))
            y_train.append(0)
            train_neg_count += 1

    if not is_train:
        val_candidates[s1_id] = cands_with_feats

print(f"Feature dataset created in {time.time()-t0:.2f}s.", flush=True)
print(f"Train Dataset: {train_pos_count:,} positives, {train_neg_count:,} hard negatives.", flush=True)
print(f"Validation Dataset: {len(val_candidates):,} S1 entities.", flush=True)

feature_names = list(feat.keys())

# 5. Train Baseline Model (HistGradientBoostingClassifier)
print("\nTraining HistGradientBoostingClassifier baseline...", flush=True)
t_train = time.time()
model = HistGradientBoostingClassifier(random_state=42, max_iter=150, min_samples_leaf=20)
model.fit(X_train, y_train)
print(f"Model fit complete in {time.time()-t_train:.2f}s.", flush=True)

# 6. Evaluate Decision Threshold Grid Search on Validation Set
print("\n=== THRESHOLD CALIBRATION GRID SEARCH (6,000 Validation S1 Entities) ===", flush=True)

val_gt_sub = {s1_id: val_gt_map[s1_id] for s1_id in val_s1_ids}

val_pairs_list = []
val_pair_index = []

for s1_id, cands in val_candidates.items():
    for tid, fvec in cands:
        val_pairs_list.append(fvec)
        val_pair_index.append((s1_id, tid))

print(f"Validation features to predict: {len(val_pairs_list):,} pairs...", flush=True)
t_pred = time.time()
if val_pairs_list:
    val_probs = model.predict_proba(val_pairs_list)[:, 1]
else:
    val_probs = np.array([])
print(f"Predictions completed in {time.time()-t_pred:.2f}s.", flush=True)

s1_candidate_probs = defaultdict(list)
for (s1_id, tid), prob in zip(val_pair_index, val_probs):
    s1_candidate_probs[s1_id].append((tid, prob))

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = []
best_thresh = None
best_f05 = -1.0

for T in thresholds:
    preds = {}
    for s1_id in val_s1_ids:
        cand_p = s1_candidate_probs.get(s1_id, [])
        matches = [tid for tid, prob in cand_p if prob >= T]
        preds[s1_id] = matches

    res = evaluate_predictions(preds, val_gt_sub)
    macro_f05 = res['macro_f0.5']
    prec = res['macro_precision']
    rec = res['macro_recall']
    singletons = res['exact_singletons_credit']
    
    print(f"Threshold T = {T:.2f} | Macro F0.5 = {macro_f05:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | Singletons = {singletons:.4f}", flush=True)
    
    results.append({
        'Threshold': T,
        'Macro F0.5': macro_f05,
        'Precision': prec,
        'Recall': rec,
        'Singletons Credit': singletons
    })
    
    if macro_f05 > best_f05:
        best_f05 = macro_f05
        best_thresh = T

df_res = pd.DataFrame(results)
print(f"\nOPTIMAL THRESHOLD: T* = {best_thresh:.2f} with Macro F0.5 = {best_f05:.4f}", flush=True)

df_res.to_csv(base_dir / "scratch" / "phase4_threshold_calibration.csv", index=False)
print("Saved threshold calibration to scratch/phase4_threshold_calibration.csv", flush=True)
