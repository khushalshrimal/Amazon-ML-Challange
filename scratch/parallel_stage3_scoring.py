"""
Multiprocess Parallel Stage 3 ML Scoring Pipeline for Amazon ML Challenge.
Reuses existing 1.08 GB output/candidate_pairs.tsv directly.
Uses Python multiprocessing across ALL available CPU cores.
Streams candidate pairs and S1 records in 5,000-S1 chunks with 8 worker processes.
Saves checkpoint after every chunk to output/checkpoint.json.
Emits mandatory live progress & heartbeats every completed chunk.
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
import multiprocessing as mp

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from inference_optimized import OnDemandTargetStore, fast_norm_name, fast_norm_addr, extract_digits, extract_ngrams
from progress_logger import format_time

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')

# Global worker variables
_w_target_attr = None
_w_model = None
_w_cache = {}

def _worker_init(target_attr_dict, model_path_str):
    global _w_target_attr, _w_model, _w_cache
    _w_target_attr = target_attr_dict
    _w_model = joblib.load(model_path_str)
    _w_cache = {}

def _get_target_parsed(mid2):
    if mid2 not in _w_cache:
        rec_str = _w_target_attr[mid2]
        c2, norm_n2, norm_a2 = rec_str.split('\t')
        _w_cache[mid2] = (
            c2, norm_n2, norm_a2,
            set(norm_n2.split()),
            set(norm_a2.split()),
            extract_ngrams(norm_n2, 3),
            extract_ngrams(norm_a2, 3),
            extract_digits(norm_a2),
            len(norm_n2),
            len(norm_a2),
            len(set(norm_n2.split())),
            len(set(norm_a2.split())),
            len(extract_ngrams(norm_n2, 3)),
            len(extract_ngrams(norm_a2, 3)),
            len(extract_digits(norm_a2)),
            1.0 if mid2.startswith('S2-') else 0.0
        )
    return _w_cache[mid2]

def _worker_score_subchunk(subchunk_tasks):
    """
    Subchunk task: list of tuples (s1_id, norm_n, norm_a, c, toks_n, toks_a, ngrams_n, ngrams_a, digits_a, cand_str)
    Returns: list of (s1_id, match_str), total_cands_evaluated, matches_found_count
    """
    results = []
    total_cands = 0
    matches_found = 0

    for s1_rec in subchunk_tasks:
        s1_id = s1_rec['s1_id']
        cands = s1_rec['cands']
        total_cands += len(cands)

        valid_tids = [tid for tid in cands if tid in _w_target_attr]

        m_list = []
        if valid_tids:
            X_batch = np.empty((len(valid_tids), 13), dtype=np.float32)
            
            for row_idx, tid in enumerate(valid_tids):
                t_parsed = _get_target_parsed(tid)
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

            probs = _w_model.predict_proba(X_batch)[:, 1]
            for tid, prob in zip(valid_tids, probs):
                if prob >= 0.98:
                    m_list.append(tid)

        matches_found += len(m_list)
        match_str = ",".join(m_list)
        results.append(f"{s1_id}\t{match_str}\n")

    return results, total_cands, matches_found

def get_ram_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def get_cpu_pct():
    return psutil.cpu_percent(interval=None)

def main():
    t0_start = time.time()
    
    print("=" * 80, flush=True)
    print("=== MULTIPROCESS PARALLEL STAGE 3 ML SCORING PIPELINE ===", flush=True)
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

    # --- STEP 2: BUILD TARGET STORE FOR WORKER INHERITANCE ---
    print("\n--- STEP 2: Loading Target Attribute Store ---", flush=True)
    t0_target = time.time()
    target_attr_dict = {}
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
                target_attr_dict[mid] = f"{c}\t{norm_n}\t{norm_a}"
                total_target_loaded += 1
            
            elapsed = time.time() - t0_target
            pct = (total_target_loaded / total_expected_targets) * 100.0
            print(f"  Target Cache: {total_target_loaded:,}/{total_expected_targets:,} ({pct:.1f}%) | RAM: {get_ram_gb():.2f} GB | Elapsed: {format_time(elapsed)}", flush=True)

    print(f"Target Attribute Store loaded across {len(target_attr_dict):,} records in {time.time() - t0_target:.2f}s.\n", flush=True)

    # --- STEP 3: CHECKPOINT & RESUME MANAGEMENT ---
    print("--- STEP 3: Checking Checkpoint State & Initializing Output Files ---", flush=True)
    completed_chunk_idx = 0
    total_s1_processed = 0
    total_candidates_evaluated = 0
    total_matches_found = 0

    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
            completed_chunk_idx = ckpt_data.get("completed_chunks", 0)
            total_s1_processed = ckpt_data.get("processed_s1", 0)
            total_candidates_evaluated = ckpt_data.get("processed_candidates", 0)
            total_matches_found = ckpt_data.get("predicted_matches", 0)
        print(f"RESUMING FROM CHECKPOINT! Already completed: {completed_chunk_idx} chunks ({total_s1_processed:,} S1 entities).", flush=True)
    else:
        print("Starting fresh Stage 3 run. Writing matching_results.tsv header...", flush=True)
        with open(match_file, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # --- STEP 4: LAUNCH MULTIPROCESS WORKER POOL ---
    num_workers = min(os.cpu_count() or 4, 8)
    total_s1_records = 1732544
    chunk_size = 5000  # 5,000 S1 records per chunk across workers
    total_chunks = (total_s1_records + chunk_size - 1) // chunk_size
    subchunk_size = 1000  # Sub-chunk per worker map task

    print(f"\n--- STAGE 3/3: Multiprocess Parallel ML Scoring ({num_workers} Workers) ---", flush=True)
    print(f"Total S1: {total_s1_records:,} | Chunk Size: {chunk_size:,} | Total Chunks: {total_chunks:,} | Threshold: 0.98\n", flush=True)

    t0_scoring_start = time.time()
    last_log_time = time.time()

    # Launch multiprocessing pool
    pool = mp.Pool(processes=num_workers, initializer=_worker_init, initargs=(target_attr_dict, str(model_path)))
    print(f"Worker pool initialized with {num_workers} active processes.", flush=True)

    # Open file streams
    s1_file = open(s1_path, "r", encoding="utf-8")
    cand_stream = open(cand_file, "r", encoding="utf-8")

    # Skip headers
    s1_file.readline()
    cand_stream.readline()

    # Fast forward if resuming
    if completed_chunk_idx > 0:
        skip_rows = completed_chunk_idx * chunk_size
        print(f"Fast-forwarding file streams past {skip_rows:,} completed S1 rows...", flush=True)
        for _ in range(skip_rows):
            s1_file.readline()
            cand_stream.readline()
        print("Streams fast-forwarded successfully.", flush=True)

    chunk_idx = completed_chunk_idx

    while total_s1_processed < total_s1_records:
        chunk_idx += 1
        t_chunk_start = time.time()

        # Read next chunk of 5,000 rows from streams
        s1_rows = []
        cand_rows = []

        for _ in range(chunk_size):
            s1_line = s1_file.readline()
            cand_line = cand_stream.readline()
            if not s1_line:
                break
            s1_rows.append(s1_line)
            cand_rows.append(cand_line)

        if not s1_rows:
            break

        # Build subchunk tasks for workers
        subchunk_tasks = []
        for i in range(0, len(s1_rows), subchunk_size):
            sub_s1 = s1_rows[i:i + subchunk_size]
            sub_cand = cand_rows[i:i + subchunk_size]
            
            task_list = []
            for s1_l, cand_l in zip(sub_s1, sub_cand):
                parts = s1_l.strip().split("\t")
                s1_id = parts[0]
                r_n = parts[1] if len(parts) > 1 else ""
                r_a = parts[2] if len(parts) > 2 else ""
                c = parts[3] if len(parts) > 3 and parts[3] else "UNKNOWN"

                c_parts = cand_l.strip().split("\t")
                cand_str = c_parts[1] if len(c_parts) > 1 else ""
                cands = cand_str.split(",") if cand_str else []

                norm_n = fast_norm_name(r_n)
                norm_a = fast_norm_addr(r_a)

                task_list.append({
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
                    'len_dig_a': len(extract_digits(norm_a)),
                    'cands': cands
                })
            subchunk_tasks.append(task_list)

        # Execute parallel workers across subchunks
        worker_results = pool.map(_worker_score_subchunk, subchunk_tasks)

        chunk_match_lines = []
        chunk_cands = 0
        chunk_matches = 0

        for lines, c_cnt, m_cnt in worker_results:
            chunk_match_lines.extend(lines)
            chunk_cands += c_cnt
            chunk_matches += m_cnt

        total_s1_processed += len(s1_rows)
        total_candidates_evaluated += chunk_cands
        total_matches_found += chunk_matches

        # Append match lines to matching_results.tsv
        with open(match_file, "a", encoding="utf-8") as f_match:
            f_match.writelines(chunk_match_lines)

        # Save Checkpoint JSON
        ckpt_data = {
            "completed_chunks": chunk_idx,
            "processed_s1": total_s1_processed,
            "processed_candidates": total_candidates_evaluated,
            "predicted_matches": total_matches_found,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
            json.dump(ckpt_data, f_ckpt, indent=2)

        # MANDATORY LIVE PROGRESS REPORT AFTER EVERY COMPLETED CHUNK
        now = time.time()
        elapsed_scoring = now - t0_scoring_start
        pct = (total_s1_processed / total_s1_records) * 100.0
        rate = total_s1_processed / elapsed_scoring if elapsed_scoring > 0 else 0
        eta = (total_s1_records - total_s1_processed) / rate if rate > 0 else 0

        print(f"[STAGE 3/3 PARALLEL SCORING]", flush=True)
        print(f"S1: {total_s1_processed:,} / {total_s1_records:,}", flush=True)
        print(f"Progress: {pct:.1f}%", flush=True)
        print(f"Candidates processed: {total_candidates_evaluated:,}", flush=True)
        print(f"Matches >= 0.98: {total_matches_found:,}", flush=True)
        print(f"Rate: {rate:,.1f} S1/sec", flush=True)
        print(f"Elapsed: {format_time(elapsed_scoring)}", flush=True)
        print(f"ETA: {format_time(eta)}", flush=True)
        print(f"Batch: {chunk_idx:,} / {total_chunks:,}", flush=True)
        print(f"Workers: {num_workers}", flush=True)
        print(f"RAM: {get_ram_gb():.2f} GB", flush=True)
        print(f"CPU: {get_cpu_pct()}%", flush=True)
        print(f"Status: ACTIVE\n", flush=True)

        del s1_rows, cand_rows, subchunk_tasks, worker_results, chunk_match_lines
        gc.collect()

    pool.close()
    pool.join()
    s1_file.close()
    cand_stream.close()

    t_scoring_total = time.time() - t0_scoring_start
    t_total = time.time() - t0_start

    print("\n" + "=" * 80, flush=True)
    print("=== STAGE 3 PARALLEL SCORING COMPLETED SUCCESSFULLY ===", flush=True)
    print("=" * 80, flush=True)
    print(f"1. Total Pipeline Runtime:      {t_total:.2f} s ({t_total/60.0:.2f} min)", flush=True)
    print(f"2. ML Parallel Scoring Time:    {t_scoring_total:.2f} s ({t_scoring_total/60.0:.2f} min)", flush=True)
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
    mp.freeze_support()
    main()
