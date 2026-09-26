"""
Standalone Resumable Stage 3 Python ML Scoring Pipeline for Amazon ML Challenge.
Reuses existing 1.08 GB output/candidate_pairs.tsv directly.
Streams S1 records and candidate pairs in small batches (250 S1 records/batch) with ZERO pre-parsing delay.
Saves checkpoint after every batch to output/checkpoint.json.
Emits mandatory live terminal progress & heartbeats.
"""

import os
import sys
import time
import re
import gc
import json
import joblib
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from inference_optimized import OnDemandTargetStore, fast_norm_name, fast_norm_addr, extract_digits, extract_ngrams
from progress_logger import format_time

def get_ram_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def get_cpu_pct():
    return psutil.cpu_percent(interval=None)

def main():
    t0_start = time.time()
    
    print("=" * 80, flush=True)
    print("=== STANDALONE RESUMABLE STAGE 3 ML SCORING PIPELINE ===", flush=True)
    print("=" * 80, flush=True)

    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"
    checkpoint_file = output_dir / "checkpoint.json"
    model_path = base_dir / "models" / "lightgbm_model.joblib"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    s1_path = test_dir / "test_source1.tsv"

    # --- STEP 1: VERIFY MANDATORY PRE-REQUISITES ---
    print("\n--- STEP 1: Verifying Mandatory Prerequisites ---", flush=True)
    if not cand_file.exists():
        print("ERROR: output/candidate_pairs.tsv does not exist!", flush=True)
        sys.exit(1)
    
    cand_size_gb = cand_file.stat().st_size / (1024 * 1024 * 1024)
    print(f"Verified output/candidate_pairs.tsv exists ({cand_size_gb:.2f} GB).", flush=True)

    if not model_path.exists():
        print(f"ERROR: Model file {model_path} does not exist!", flush=True)
        sys.exit(1)
    print(f"Verified LightGBM model artifact exists at {model_path.name}.", flush=True)

    model = joblib.load(model_path)
    print("Loaded LightGBM model successfully.", flush=True)

    # --- STEP 2: BUILD TARGET STORE FOR FEATURE EXTRACTION ---
    print("\n--- STEP 2: Loading Target Attribute Store ---", flush=True)
    t0_target = time.time()
    target_store = OnDemandTargetStore()
    total_target_loaded = 0
    total_expected_targets = 9969589

    for fpath, label in [(s2_path, "Source-2"), (s3_path, "Source-3")]:
        print(f"Indexing target attributes from {label} ({fpath.name})...", flush=True)
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=200000, dtype=str):
            eids = chunk['entity_id'].values
            raw_names = chunk['business_name'].values
            raw_addrs = chunk['business_address'].values
            countries = chunk['country'].fillna('UNKNOWN').astype(str).values
            
            for mid, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
                norm_n = fast_norm_name(r_n)
                norm_a = fast_norm_addr(r_a)
                target_store.add_target_raw(mid, c, norm_n, norm_a)
                total_target_loaded += 1
            
            elapsed = time.time() - t0_target
            pct = (total_target_loaded / total_expected_targets) * 100.0
            print(f"  Target Cache: {total_target_loaded:,}/{total_expected_targets:,} ({pct:.1f}%) | RAM: {get_ram_gb():.2f} GB | Elapsed: {format_time(elapsed)}", flush=True)

    print(f"Target Attribute Store loaded across {len(target_store):,} records in {time.time() - t0_target:.2f}s.\n", flush=True)

    # --- STEP 3: CHECKPOINT & RESUME MANAGEMENT ---
    print("--- STEP 3: Checking Checkpoint State & Initializing Output Files ---", flush=True)
    completed_batch_idx = 0
    total_s1_processed = 0
    total_candidates_evaluated = 0
    total_matches_found = 0

    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
            completed_batch_idx = ckpt_data.get("completed_batches", 0)
            total_s1_processed = ckpt_data.get("processed_s1", 0)
            total_candidates_evaluated = ckpt_data.get("processed_candidates", 0)
            total_matches_found = ckpt_data.get("predicted_matches", 0)
        print(f"RESUMING FROM CHECKPOINT! Already completed: {completed_batch_idx} batches ({total_s1_processed:,} S1 entities).", flush=True)
    else:
        print("Starting fresh Stage 3 run. Writing matching_results.tsv header...", flush=True)
        with open(match_file, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # --- STEP 4: STREAMING STAGE 3 BATCH FEATURE EXTRACTION & SCORING ---
    total_s1_records = 1732544
    batch_size = 250  # Small S1 batch size per user prompt
    total_batches = (total_s1_records + batch_size - 1) // batch_size

    print(f"\n--- STAGE 3/3: Streaming LightGBM Feature Extraction & Scoring ---", flush=True)
    print(f"Total S1: {total_s1_records:,} | Batch Size: {batch_size} | Total Batches: {total_batches:,} | Threshold: 0.98\n", flush=True)

    t0_scoring_start = time.time()
    last_log_time = time.time()

    # Open both S1 query TSV and candidate_pairs.tsv synchronously as file streams
    s1_file = open(s1_path, "r", encoding="utf-8")
    cand_stream = open(cand_file, "r", encoding="utf-8")

    # Skip headers
    s1_file.readline()
    cand_stream.readline()

    # Fast forward streams if resuming from checkpoint
    if completed_batch_idx > 0:
        skip_rows = completed_batch_idx * batch_size
        print(f"Fast-forwarding file streams past {skip_rows:,} completed S1 rows...", flush=True)
        for _ in range(skip_rows):
            s1_file.readline()
            cand_stream.readline()
        print("Streams fast-forwarded successfully.", flush=True)

    batch_idx = completed_batch_idx

    while total_s1_processed < total_s1_records:
        batch_idx += 1
        t_batch_start = time.time()

        # Read next batch of 250 rows from streams
        s1_batch_rows = []
        cand_batch_rows = []

        for _ in range(batch_size):
            s1_line = s1_file.readline()
            cand_line = cand_stream.readline()
            if not s1_line:
                break
            s1_batch_rows.append(s1_line)
            cand_batch_rows.append(cand_line)

        if not s1_batch_rows:
            break

        # Process S1 batch rows
        s1_records = []
        for line in s1_batch_rows:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            r_n = parts[1] if len(parts) > 1 else ""
            r_a = parts[2] if len(parts) > 2 else ""
            c = parts[3] if len(parts) > 3 and parts[3] else "UNKNOWN"

            norm_n = fast_norm_name(r_n)
            norm_a = fast_norm_addr(r_a)

            s1_records.append({
                's1_id': s1_id,
                'norm_n': norm_n,
                'norm_a': norm_a,
                'toks_n': set(norm_n.split()),
                'toks_a': set(norm_a.split()),
                'ngrams_n': extract_ngrams(norm_n, 3),
                'ngrams_a': extract_ngrams(norm_a, 3),
                'digits_a': extract_digits(norm_a),
                'c': c,
                'len_n': len(norm_n),
                'len_a': len(norm_a),
                'len_toks_n': len(set(norm_n.split())),
                'len_toks_a': len(set(norm_a.split())),
                'len_ng_n': len(extract_ngrams(norm_n, 3)),
                'len_ng_a': len(extract_ngrams(norm_a, 3)),
                'len_dig_a': len(extract_digits(norm_a))
            })

        # Process candidate_pairs rows for this batch
        pair_meta = []
        batch_cand_count = 0

        for s1_rec, cand_line in zip(s1_records, cand_batch_rows):
            parts = cand_line.strip().split("\t")
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                cands = cand_str.split(",")
                batch_cand_count += len(cands)
                for tid in cands:
                    if tid in target_store.raw_attr:
                        pair_meta.append((s1_rec, tid))

        total_candidates_evaluated += batch_cand_count

        # Feature Extraction & Model Scoring
        match_lines_dict = defaultdict(list)
        batch_matches = 0

        if pair_meta:
            X_batch = np.empty((len(pair_meta), 13), dtype=np.float32)
            
            for row_idx, (s1_rec, tid) in enumerate(pair_meta):
                t_parsed = target_store.get_parsed(tid)
                c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = t_parsed[0], t_parsed[1], t_parsed[2], t_parsed[3], t_parsed[4], t_parsed[5], t_parsed[6], t_parsed[7]
                len_n2, len_a2, len_tn2, len_ta2, len_ng2_n, len_ng2_a, len_d2, is_s2 = t_parsed[8], t_parsed[9], t_parsed[10], t_parsed[11], t_parsed[12], t_parsed[13], t_parsed[14], t_parsed[15]

                n_exact = 1.0 if s1_rec['norm_n'] == norm_n2 and s1_rec['norm_n'] != "" else 0.0
                
                len_tn1 = s1_rec['len_toks_n']
                if len_tn1 > 0 and len_tn2 > 0:
                    n_overlap = len(s1_rec['toks_n'] & toks_n2)
                    n_jaccard = n_overlap / (len_tn1 + len_tn2 - n_overlap)
                else: n_overlap, n_jaccard = 0, 0.0

                len_ng1_n = s1_rec['len_ng_n']
                if len_ng1_n > 0 and len_ng2_n > 0:
                    ng_n_overlap = len(s1_rec['ngrams_n'] & ngrams_n2)
                    n_char_jaccard = ng_n_overlap / (len_ng1_n + len_ng2_n - ng_n_overlap)
                else: n_char_jaccard = 0.0

                len_ta1 = s1_rec['len_toks_a']
                if len_ta1 > 0 and len_ta2 > 0:
                    a_overlap = len(s1_rec['toks_a'] & toks_a2)
                    a_jaccard = a_overlap / (len_ta1 + len_ta2 - a_overlap)
                else: a_overlap, a_jaccard = 0, 0.0

                len_ng1_a = s1_rec['len_ng_a']
                if len_ng1_a > 0 and len_ng2_a > 0:
                    ng_a_overlap = len(s1_rec['ngrams_a'] & ngrams_a2)
                    a_char_jaccard = ng_a_overlap / (len_ng1_a + len_ng2_a - ng_a_overlap)
                else: a_char_jaccard = 0.0

                len_d1 = s1_rec['len_dig_a']
                if len_d1 > 0 and len_d2 > 0:
                    d_overlap = len(s1_rec['digits_a'] & digits_a2)
                    d_jaccard = d_overlap / (len_d1 + len_d2 - d_overlap)
                else: d_jaccard = 0.5

                c_match = 1.0 if s1_rec['c'] == c2 and s1_rec['c'] != "" else 0.0
                len_diff_n = float(abs(s1_rec['len_n'] - len_n2))
                len_diff_a = float(abs(s1_rec['len_a'] - len_a2))
                missing_addr = 1.0 if s1_rec['len_a'] == 0 or len_a2 == 0 else 0.0

                X_batch[row_idx, 0] = n_exact
                X_batch[row_idx, 1] = n_jaccard
                X_batch[row_idx, 2] = n_char_jaccard
                X_batch[row_idx, 3] = a_jaccard
                X_batch[row_idx, 4] = a_char_jaccard
                X_batch[row_idx, 5] = d_jaccard
                X_batch[row_idx, 6] = c_match
                X_batch[row_idx, 7] = len_diff_n
                X_batch[row_idx, 8] = len_diff_a
                X_batch[row_idx, 9] = float(n_overlap)
                X_batch[row_idx, 10] = float(a_overlap)
                X_batch[row_idx, 11] = missing_addr
                X_batch[row_idx, 12] = is_s2

            probs = model.predict_proba(X_batch)[:, 1]
            for (s1_rec, tid), prob in zip(pair_meta, probs):
                if prob >= 0.98:
                    match_lines_dict[s1_rec['s1_id']].append(tid)

        # Prepare match output lines
        match_out_lines = []
        for s1_rec in s1_records:
            s1_id = s1_rec['s1_id']
            m_list = match_lines_dict.get(s1_id, [])
            if m_list:
                batch_matches += len(m_list)
                match_str = ",".join(m_list)
            else:
                match_str = ""
            match_out_lines.append(f"{s1_id}\t{match_str}\n")

        total_matches_found += batch_matches
        total_s1_processed += len(s1_batch_rows)

        # Append to matching_results.tsv
        with open(match_file, "a", encoding="utf-8") as f_match:
            f_match.writelines(match_out_lines)

        # Write Checkpoint JSON
        ckpt_data = {
            "completed_batches": batch_idx,
            "processed_s1": total_s1_processed,
            "processed_candidates": total_candidates_evaluated,
            "predicted_matches": total_matches_found,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
            json.dump(ckpt_data, f_ckpt, indent=2)

        # LIVE PROGRESS LOGGING & HEARTBEAT (Logged every 2 seconds or 2% completion)
        now = time.time()
        if (now - last_log_time) >= 2.0 or total_s1_processed == total_s1_records:
            last_log_time = now
            elapsed_scoring = now - t0_scoring_start
            pct = (total_s1_processed / total_s1_records) * 100.0
            rate = total_s1_processed / elapsed_scoring if elapsed_scoring > 0 else 0
            eta = (total_s1_records - total_s1_processed) / rate if rate > 0 else 0

            print(f"[STAGE 3/3]", flush=True)
            print(f"S1: {total_s1_processed:,} / {total_s1_records:,}", flush=True)
            print(f"Progress: {pct:.1f}%", flush=True)
            print(f"Candidates processed: {total_candidates_evaluated:,}", flush=True)
            print(f"Matches >= 0.98: {total_matches_found:,}", flush=True)
            print(f"Rate: {rate:,.1f} S1/sec", flush=True)
            print(f"Elapsed: {format_time(elapsed_scoring)}", flush=True)
            print(f"ETA: {format_time(eta)}", flush=True)
            print(f"Batch: {batch_idx:,} / {total_batches:,}", flush=True)
            print(f"RAM: {get_ram_gb():.2f} GB", flush=True)
            print(f"CPU: {get_cpu_pct()}%", flush=True)
            print(f"Status: ACTIVE\n", flush=True)

        del s1_batch_rows, cand_batch_rows, s1_records, pair_meta, match_lines_dict, match_out_lines
        if 'X_batch' in locals(): del X_batch
        gc.collect()

    s1_file.close()
    cand_stream.close()

    t_scoring_total = time.time() - t0_scoring_start
    t_total = time.time() - t0_start

    print("\n" + "=" * 80, flush=True)
    print("=== STAGE 3 ML SCORING COMPLETED SUCCESSFULLY ===", flush=True)
    print("=" * 80, flush=True)
    print(f"1. Total Pipeline Runtime:      {t_total:.2f} s ({t_total/60.0:.2f} min)", flush=True)
    print(f"2. ML Scoring Runtime:          {t_scoring_total:.2f} s ({t_scoring_total/60.0:.2f} min)", flush=True)
    print(f"3. Total S1 Records Processed: {total_s1_processed:,}", flush=True)
    print(f"4. Total Candidates Evaluated: {total_candidates_evaluated:,}", flush=True)
    print(f"5. Total Matches Found (T>=0.98): {total_matches_found:,}", flush=True)
    print(f"6. Final Peak RAM:             {get_ram_gb():.2f} GB", flush=True)

    cand_bytes = cand_file.stat().st_size if cand_file.exists() else 0
    match_bytes = match_file.stat().st_size if match_file.exists() else 0
    print(f"7. output/candidate_pairs.tsv:  {cand_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"8. output/matching_results.tsv: {match_bytes / (1024*1024):.2f} MB", flush=True)
    print("=" * 80 + "\n", flush=True)

    # --- STEP 5: RUN OFFICIAL SUBMISSION VALIDATOR ---
    print("--- STEP 5: Running Official Submission Validator ---", flush=True)
    import subprocess
    validator_script = base_dir / "utils" / "validate_submission.py"
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(test_dir)
    ]

    print("Running Official Validator:", " ".join(cmd), flush=True)
    val_res = subprocess.run(cmd, capture_output=True, text=True)
    val_output = val_res.stdout + "\n" + val_res.stderr
    print("=" * 80, flush=True)
    print("OFFICIAL VALIDATOR OUTPUT:\n", val_output, flush=True)
    print("=" * 80, flush=True)

    if val_res.returncode == 0:
        print("\nSUBMISSION VALIDATION: PASSED SUCCESSFULLY!", flush=True)
    else:
        print("\nSUBMISSION VALIDATION: FAILED WITH ERRORS!", flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
