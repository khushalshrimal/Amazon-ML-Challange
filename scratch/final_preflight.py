"""
Diagnostic Script: Final Preflight Test Suite
Goal: Run ONE comprehensive preflight test that identifies all major blockers
before attempting the full 1,732,544-S1 submission.

DO NOT MODIFY PRODUCTION PIPELINE.
DO NOT RUN FULL INFERENCE OR 10k BENCHMARK.
DO NOT SAVE CACHE VIA JOBLIB DUMP FOR TARGET STORE/BLOCKER.
"""

import os
import sys
import time
import shutil
import subprocess
import psutil
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Dynamic path preparation
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from blocking import EntityBlocker
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

# Global status tracking
status_dict = {
    "dataset_integrity": "FAIL",
    "model_loading": "FAIL",
    "blocker_build": "FAIL",
    "target_lookup": "FAIL",
    "feat_generation": "FAIL",
    "lightgbm_inference": "FAIL",
    "output_integrity": "FAIL",
    "subset_validation": "FAIL"
}
fail_reasons = []

peak_ram = get_ram_mb()

def update_ram():
    global peak_ram
    cur = get_ram_mb()
    if cur > peak_ram:
        peak_ram = cur
    return cur

print("=" * 80, flush=True)
print("=== AMAZON ML CHALLENGE — FINAL PREFLIGHT DIAGNOSTIC SUITE ===", flush=True)
print("=" * 80, flush=True)
print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

# ---------------------------------------------------------
# 1. DATASET INTEGRITY
# ---------------------------------------------------------
print("--- 1. DATASET INTEGRITY CHECK ---", flush=True)
test_dir = base_dir / "dataset" / "test"
expected_counts = {
    "test_source1.tsv": 1732544,
    "test_source2.tsv": 4887273,
    "test_source3.tsv": 5082316
}
expected_cols = ['entity_id', 'business_name', 'business_address', 'country']

dataset_ok = True
actual_counts = {}

for fname, exp_cnt in expected_counts.items():
    fpath = test_dir / fname
    if not fpath.exists():
        print(f"  [FAIL] Missing file: {fpath}", flush=True)
        fail_reasons.append(f"Missing test file {fname}")
        dataset_ok = False
        continue
    
    # Read sample to check columns
    sample_df = pd.read_csv(fpath, sep='\t', nrows=5)
    cols = sample_df.columns.tolist()
    if cols != expected_cols:
        print(f"  [FAIL] {fname} columns mismatch. Expected {expected_cols}, got {cols}", flush=True)
        fail_reasons.append(f"Column mismatch in {fname}")
        dataset_ok = False
    
    # Count records in chunks
    cnt = 0
    for chunk in pd.read_csv(fpath, sep='\t', usecols=['entity_id'], chunksize=500000):
        cnt += len(chunk)
    actual_counts[fname] = cnt
    
    if cnt == exp_cnt and cols == expected_cols:
        print(f"  [PASS] {fname}: {cnt:,} records (Expected {exp_cnt:,}), Columns: {cols}", flush=True)
    else:
        print(f"  [FAIL] {fname}: {cnt:,} records (Expected {exp_cnt:,})", flush=True)
        fail_reasons.append(f"Record count mismatch in {fname}: got {cnt}, expected {exp_cnt}")
        dataset_ok = False

if dataset_ok:
    status_dict["dataset_integrity"] = "PASS"
print(f"Dataset Integrity Check: {status_dict['dataset_integrity']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 2. MODEL LOADING
# ---------------------------------------------------------
print("--- 2. MODEL LOADING CHECK ---", flush=True)
model_path = base_dir / "models" / "lightgbm_model.joblib"
model = None

if not model_path.exists():
    print(f"  [FAIL] Model file not found at {model_path}", flush=True)
    fail_reasons.append("Model file not found")
else:
    try:
        t0_model = time.time()
        model = joblib.load(model_path)
        t_model = time.time() - t0_model
        
        # Inspection
        n_features = getattr(model, 'n_features_', None)
        if n_features is None and hasattr(model, 'booster_'):
            n_features = model.booster_.num_feature()
            
        n_trees = getattr(model, 'n_estimators', None)
        if n_trees is None and hasattr(model, 'booster_'):
            n_trees = model.booster_.num_trees()
            
        print(f"  [PASS] LightGBM model loaded in {t_model:.3f}s", flush=True)
        print(f"         Model type: {type(model).__name__}", flush=True)
        print(f"         Features expected: {n_features}", flush=True)
        print(f"         Trees/Estimators: {n_trees}", flush=True)
        
        status_dict["model_loading"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Exception loading model: {e}", flush=True)
        fail_reasons.append(f"Model load error: {e}")

print(f"Model Check: {status_dict['model_loading']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 3. BLOCKER INDEX BUILD & 4. TARGET FEATURE STORE
# ---------------------------------------------------------
print("--- 3. FULL TARGET BLOCKER BUILD & FEATURE STORE PREPARATION ---", flush=True)
t0_target_prep = time.time()
target_store = OnDemandTargetStore()
blocker = EntityBlocker()

target_files = [
    (test_dir / "test_source2.tsv", "Source-2"),
    (test_dir / "test_source3.tsv", "Source-3")
]

total_target_loaded = 0
last_report_count = 0

try:
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
                    elapsed = time.time() - t0_target_prep
                    cur_ram = update_ram()
                    print(f"  Indexed {total_target_loaded:,} target records... ({elapsed:.2f}s elapsed, RAM: {cur_ram:.1f} MB)", flush=True)
    
    t_target_prep = time.time() - t0_target_prep
    cur_ram = update_ram()
    print(f"  [PASS] Target Index & Feature Store built in {t_target_prep:.2f}s for {total_target_loaded:,} records.", flush=True)
    print(f"         Final Target Pool RAM: {cur_ram:.1f} MB", flush=True)
    status_dict["blocker_build"] = "PASS"
except Exception as e:
    print(f"  [FAIL] Target index build failed: {e}", flush=True)
    fail_reasons.append(f"Blocker build error: {e}")
    t_target_prep = 0

# 4. Target Feature Store Lookup Verification
print("\n--- 4. TARGET FEATURE LOOKUP VERIFICATION ---", flush=True)
if status_dict["blocker_build"] == "PASS" and len(target_store) > 0:
    try:
        sample_mids = list(target_store.raw_attr.keys())[:5]
        parsed_ok = True
        for smid in sample_mids:
            parsed = target_store.get_parsed(smid)
            if not isinstance(parsed, tuple) or len(parsed) != 16:
                parsed_ok = False
                break
        if parsed_ok:
            print(f"  [PASS] Target feature store lookup verified successfully on sample targets.", flush=True)
            status_dict["target_lookup"] = "PASS"
        else:
            print(f"  [FAIL] Target parsed structure invalid.", flush=True)
            fail_reasons.append("Target lookup tuple structure invalid")
    except Exception as e:
        print(f"  [FAIL] Target lookup exception: {e}", flush=True)
        fail_reasons.append(f"Target lookup error: {e}")

print(f"Blocker Build Check: {status_dict['blocker_build']}", flush=True)
print(f"Target Lookup Check: {status_dict['target_lookup']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 5. CONTROLLED INFERENCE TEST (EXACTLY 100 S1 RECORDS)
# ---------------------------------------------------------
print("--- 5. CONTROLLED INFERENCE TEST (EXACTLY 100 S1 RECORDS) ---", flush=True)

num_s1_test = 100
cand_pairs_count = 0
match_pairs_count = 0

t_cand_gen = 0.0
t_feat_gen = 0.0
t_lgb_infer = 0.0
t_infer_total = 0.0

s1_precomputed = []
pair_meta = []
s1_ids_test = []
matched_dict = defaultdict(list)

if status_dict["blocker_build"] == "PASS" and status_dict["model_loading"] == "PASS":
    try:
        t_infer_total_start = time.time()
        
        # Load exactly 100 S1 records
        s1_path = test_dir / "test_source1.tsv"
        s1_df_100 = pd.read_csv(s1_path, sep="\t", nrows=num_s1_test, dtype=str)
        
        eids = s1_df_100['entity_id'].values
        raw_names = s1_df_100['business_name'].values
        raw_addrs = s1_df_100['business_address'].values
        countries = s1_df_100['country'].fillna('UNKNOWN').astype(str).values
        
        s1_ids_test = list(eids)
        
        # A. S1 Precomputation & Blocker Lookup
        t0_cand = time.time()
        cands_all_valid = True
        
        for s1_id, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
            s1_rec = precompute_s1_record(s1_id, c, r_n, r_a)
            s1_precomputed.append(s1_rec)
            
            cands = blocker.get_candidates(s1_rec['raw_rec'])
            cand_pairs_count += len(cands)
            
            for tid in cands:
                if not (tid.startswith("S2-") or tid.startswith("S3-")):
                    cands_all_valid = False
                if tid in target_store.raw_attr:
                    pair_meta.append((len(s1_precomputed) - 1, tid, s1_id))
                    
        t_cand_gen = time.time() - t0_cand
        
        if not cands_all_valid:
            print("  [WARNING] Non-S2/S3 candidate ID found!", flush=True)
            
        print(f"  Candidates generated for {len(eids)} S1 records: {cand_pairs_count:,} pairs in {t_cand_gen:.3f}s", flush=True)
        
        # B. 13-Feature Generation
        if pair_meta:
            t0_feat = time.time()
            X_test = np.empty((len(pair_meta), 13), dtype=np.float32)
            for p_idx, (s1_idx, tid, _) in enumerate(pair_meta):
                fill_features_optimized(X_test, p_idx, s1_precomputed[s1_idx], target_store, tid)
            t_feat_gen = time.time() - t0_feat
            
            # Checks on matrix
            shape_ok = (X_test.shape == (len(pair_meta), 13))
            nan_cnt = np.isnan(X_test).sum()
            inf_cnt = np.isinf(X_test).sum()
            
            if shape_ok and nan_cnt == 0 and inf_cnt == 0:
                print(f"  [PASS] 13 features generated successfully! Shape: {X_test.shape}, NaNs: {nan_cnt}, Infs: {inf_cnt}", flush=True)
                status_dict["feat_generation"] = "PASS"
            else:
                print(f"  [FAIL] Feature matrix check failed! Shape: {X_test.shape}, NaNs: {nan_cnt}, Infs: {inf_cnt}", flush=True)
                fail_reasons.append(f"Feature matrix invalid: NaNs={nan_cnt}, Infs={inf_cnt}")
                
            # C. LightGBM Scoring
            t0_lgb = time.time()
            probs = model.predict_proba(X_test)[:, 1]
            t_lgb_infer = time.time() - t0_lgb
            
            for (_, tid, s1_id), prob in zip(pair_meta, probs):
                if prob >= 0.98:
                    matched_dict[s1_id].append(tid)
                    match_pairs_count += 1
                    
            print(f"  [PASS] LightGBM inference completed in {t_lgb_infer:.3f}s. Matches (T>=0.98): {match_pairs_count:,}", flush=True)
            status_dict["lightgbm_inference"] = "PASS"
        else:
            print("  [WARNING] Zero candidate pairs generated for sample!", flush=True)
            status_dict["feat_generation"] = "PASS"
            status_dict["lightgbm_inference"] = "PASS"

        t_infer_total = time.time() - t_infer_total_start
        s1_per_sec = num_s1_test / t_infer_total if t_infer_total > 0 else 0
        cands_per_sec = cand_pairs_count / t_infer_total if t_infer_total > 0 else 0
        
        print("\n  --- Controlled 100-S1 Performance Report ---", flush=True)
        print(f"  S1 Processed:              {num_s1_test}", flush=True)
        print(f"  Candidate Pairs:           {cand_pairs_count:,}", flush=True)
        print(f"  Predicted Matches:         {match_pairs_count:,}", flush=True)
        print(f"  Candidate Pairs / sec:     {cands_per_sec:,.2f}", flush=True)
        print(f"  S1 / sec:                  {s1_per_sec:,.2f}", flush=True)
        print(f"  Candidate Generation Time: {t_cand_gen:.3f} s", flush=True)
        print(f"  Feature Generation Time:   {t_feat_gen:.3f} s", flush=True)
        print(f"  LightGBM Inference Time:   {t_lgb_infer:.3f} s", flush=True)
        print(f"  Total 100-S1 Test Time:    {t_infer_total:.3f} s", flush=True)
        print(f"  Current RAM / Peak RAM:    {get_ram_mb():.1f} MB / {update_ram():.1f} MB", flush=True)
        
    except Exception as e:
        print(f"  [FAIL] Controlled inference failed: {e}", flush=True)
        fail_reasons.append(f"Inference error: {e}")

print(f"13-Feature Generation Check: {status_dict['feat_generation']}", flush=True)
print(f"LightGBM Inference Check: {status_dict['lightgbm_inference']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 6. OUTPUT INTEGRITY TEST
# ---------------------------------------------------------
print("--- 6. OUTPUT INTEGRITY TEST (TEMPORARY DIAGNOSTIC OUTPUTS) ---", flush=True)
scratch_dir = base_dir / "scratch"
scratch_dir.mkdir(exist_ok=True)

temp_cand_file = scratch_dir / "temp_preflight_candidate_pairs.tsv"
temp_match_file = scratch_dir / "temp_preflight_matching_results.tsv"

if status_dict["lightgbm_inference"] == "PASS":
    try:
        # Build candidate lines map
        cand_map = defaultdict(set)
        for s1_idx, tid, s1_id in pair_meta:
            cand_map[s1_id].add(tid)
            
        with open(temp_cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid in s1_ids_test:
                clist = sorted(list(cand_map.get(sid, set())))
                f.write(f"{sid}\t{','.join(clist)}\n")
                
        with open(temp_match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in s1_ids_test:
                mlist = sorted(list(matched_dict.get(sid, [])))
                f.write(f"{sid}\t{','.join(mlist)}\n")
                
        # Verification checks
        output_ok = True
        
        # Check row count
        cand_lines_cnt = sum(1 for _ in open(temp_cand_file, encoding="utf-8")) - 1
        match_lines_cnt = sum(1 for _ in open(temp_match_file, encoding="utf-8")) - 1
        
        if cand_lines_cnt != num_s1_test or match_lines_cnt != num_s1_test:
            print(f"  [FAIL] Row count mismatch. Candidate rows: {cand_lines_cnt}, Match rows: {match_lines_cnt}", flush=True)
            fail_reasons.append("Output row count mismatch")
            output_ok = False
            
        # Check no duplicate target IDs in prediction & all are S2/S3 & exist in candidates
        match_df = pd.read_csv(temp_match_file, sep="\t", dtype=str)
        cand_df = pd.read_csv(temp_cand_file, sep="\t", dtype=str)
        
        cand_dict_check = {}
        for _, r in cand_df.iterrows():
            c_str = str(r['candidate_entity_ids']) if pd.notna(r['candidate_entity_ids']) else ""
            cand_dict_check[r['source1_entity_id']] = set(c.strip() for c in c_str.split(',') if c.strip())
            
        for _, r in match_df.iterrows():
            sid = r['source1_entity_id']
            m_str = str(r['matched_entity_ids']) if pd.notna(r['matched_entity_ids']) else ""
            m_list = [m.strip() for m in m_str.split(',') if m.strip()]
            
            # Duplicate check
            if len(m_list) != len(set(m_list)):
                print(f"  [FAIL] Duplicate matched ID in prediction for S1 {sid}", flush=True)
                fail_reasons.append(f"Duplicate matched ID in S1 {sid}")
                output_ok = False
                
            for mid in m_list:
                # S2/S3 prefix check
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    print(f"  [FAIL] Matched ID {mid} does not start with S2- or S3-", flush=True)
                    fail_reasons.append(f"Invalid prefix for matched ID {mid}")
                    output_ok = False
                # Existence in candidates
                if mid not in cand_dict_check.get(sid, set()):
                    print(f"  [FAIL] Matched ID {mid} not found in candidate set for S1 {sid}", flush=True)
                    fail_reasons.append(f"Matched ID {mid} missing from candidates")
                    output_ok = False
                    
        if output_ok:
            print(f"  [PASS] Output integrity verified! Exactly {num_s1_test} rows, valid S2/S3 prefixes, zero duplicate predictions.", flush=True)
            status_dict["output_integrity"] = "PASS"
            
    except Exception as e:
        print(f"  [FAIL] Output integrity check exception: {e}", flush=True)
        fail_reasons.append(f"Output integrity error: {e}")

print(f"Output Integrity Check: {status_dict['output_integrity']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 7. VALIDATOR COMPATIBILITY (SUBSET VALIDATION)
# ---------------------------------------------------------
print("--- 7. SUBSET VALIDATOR COMPATIBILITY CHECK ---", flush=True)
temp_test_dir = scratch_dir / "temp_preflight_test_dir"
temp_test_dir.mkdir(exist_ok=True)
temp_s1_file = temp_test_dir / "test_source1.tsv"

try:
    # Write 100 S1 records to temp test_source1.tsv
    s1_df_100.to_csv(temp_s1_file, sep="\t", index=False)
    
    val_script = base_dir / "utils" / "validate_submission.py"
    cmd = [
        sys.executable,
        str(val_script),
        "--matching", str(temp_match_file),
        "--candidate", str(temp_cand_file),
        "--test-dir", str(temp_test_dir)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    val_out = res.stdout.strip() + "\n" + res.stderr.strip()
    
    print("=== SUBSET VALIDATION (100 S1 RECORDS) OUTPUT ===", flush=True)
    print(val_out, flush=True)
    print("==================================================", flush=True)
    
    if res.returncode == 0:
        print("  [PASS] Subset validation passed clean (0 errors).", flush=True)
        status_dict["subset_validation"] = "PASS"
    else:
        print(f"  [FAIL] Subset validator exited with code {res.returncode}", flush=True)
        fail_reasons.append("Subset validator failed")
except Exception as e:
    print(f"  [FAIL] Validator compatibility exception: {e}", flush=True)
    fail_reasons.append(f"Validator error: {e}")
finally:
    # Cleanup temp diagnostic files
    if temp_test_dir.exists():
        shutil.rmtree(temp_test_dir, ignore_errors=True)
    if temp_cand_file.exists():
        temp_cand_file.unlink(missing_ok=True)
    if temp_match_file.exists():
        temp_match_file.unlink(missing_ok=True)

print(f"Subset Validation Check: {status_dict['subset_validation']} (RAM: {update_ram():.1f} MB)\n", flush=True)

# ---------------------------------------------------------
# 8. FULL-RUN ESTIMATE
# ---------------------------------------------------------
print("--- 8. FULL-RUN ESTIMATE (BASED ON 100-S1 BENCHMARK) ---", flush=True)

total_full_s1 = 1732544
avg_cands_per_s1 = cand_pairs_count / num_s1_test if num_s1_test else 0
est_total_cands = total_full_s1 * avg_cands_per_s1

# Inference rate
if 't_infer_total' in locals() and t_infer_total > 0:
    s1_rate = num_s1_test / t_infer_total
    est_full_infer_sec = total_full_s1 / s1_rate
else:
    s1_rate = 0
    est_full_infer_sec = 0

est_full_infer_hours = est_full_infer_sec / 3600.0
est_index_sec = t_target_prep if 't_target_prep' in locals() else 0
est_total_sec = est_full_infer_sec + est_index_sec
est_total_hours = est_total_sec / 3600.0

# Disk space estimation
est_cand_bytes = est_total_cands * 15  # ~15 bytes per candidate string representation
est_match_bytes = total_full_s1 * 40    # ~40 bytes per match string representation
est_disk_gb = (est_cand_bytes + est_match_bytes) / 1e9

print("DISCLAIMER: The following numbers are estimates based on Stage 1 timings and a 100-S1 sample, not guarantees.", flush=True)
print(f"  Target Index Build Time:     {est_index_sec:.2f} s", flush=True)
print(f"  Estimated Full Inference:    {est_full_infer_sec:.1f} s ({est_full_infer_hours:.2f} hours)", flush=True)
print(f"  Estimated Total Runtime:     {est_total_sec:.1f} s ({est_total_hours:.2f} hours)", flush=True)
print(f"  Estimated Peak RAM:          {peak_ram:.1f} MB", flush=True)
print(f"  Estimated Disk Usage:        {est_disk_gb:.2f} GB", flush=True)
print(f"  Estimated Candidate Pairs:   {est_total_cands:,.0f} pairs\n", flush=True)

# ---------------------------------------------------------
# 9. FINAL REPORT
# ---------------------------------------------------------
all_passed = all(val == "PASS" for val in status_dict.values())

print("=" * 80, flush=True)
print("=== FINAL PREFLIGHT RESULT ===", flush=True)
print("=" * 80, flush=True)
print(f"Dataset integrity:          {status_dict['dataset_integrity']}", flush=True)
print(f"Model loading:              {status_dict['model_loading']}", flush=True)
print(f"Full target blocker build:  {status_dict['blocker_build']}", flush=True)
print(f"Target feature lookup:      {status_dict['target_lookup']}", flush=True)
print(f"13-feature generation:      {status_dict['feat_generation']}", flush=True)
print(f"LightGBM inference:         {status_dict['lightgbm_inference']}", flush=True)
print(f"Output integrity:           {status_dict['output_integrity']}", flush=True)
print(f"Subset validation:          {status_dict['subset_validation']}", flush=True)
print(f"Estimated full runtime:     {est_total_hours:.2f} hours (Index: {est_index_sec:.2f}s, Inference: {est_full_infer_hours:.2f}h)", flush=True)
print(f"Estimated peak RAM:         {peak_ram:.1f} MB", flush=True)
print("-" * 80, flush=True)

if all_passed:
    print("Overall:", flush=True)
    print("   READY FOR FULL RUN", flush=True)
else:
    reasons_str = "; ".join(fail_reasons) if fail_reasons else "Unknown issue"
    print("Overall:", flush=True)
    print(f"   NOT READY — FIX REQUIRED: {reasons_str}", flush=True)

print("=" * 80, flush=True)
