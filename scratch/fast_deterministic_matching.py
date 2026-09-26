"""
Fast Deterministic Entity Matching & Submission Generator for Amazon ML Challenge.
Generates complete output/matching_results.tsv from existing output/candidate_pairs.tsv.
Fulfills all prompt rules:
- Streaming / chunked execution with safe memory footprint.
- Live progress logs every 5-10 seconds.
- Deterministic multi-signal field matching (Name exact/Jaccard, Address Jaccard, Digits, Country).
- Automatic execution of official validator upon completion.
"""

import os
import sys
import time
import re
import gc
import json
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from progress_logger import format_time

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')
sub_digits = re.compile(r'\b\d+\b')

def fast_norm_name(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = sub_amp.sub(' and ', text)
    text = sub_punct.sub(' ', text)
    return sub_space.sub(' ', text).strip()

def fast_norm_addr(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = sub_punct.sub(' ', text)
    return sub_space.sub(' ', text).strip()

def extract_digits_set(text: str) -> set:
    return set(sub_digits.findall(text))

# Global target store per worker process
_w_target_raw = None
_w_target_parsed_cache = None

def _worker_init(s2_path_str, s3_path_str):
    global _w_target_raw, _w_target_parsed_cache
    _w_target_raw = {}
    _w_target_parsed_cache = {}
    
    for fpath in [s2_path_str, s3_path_str]:
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline() # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    eid, r_n, r_a, c = parts[0], parts[1], parts[2], parts[3]
                    _w_target_raw[eid] = (c if c != "UNKNOWN" else "", fast_norm_name(r_n), fast_norm_addr(r_a))

def _get_parsed_target(tid):
    if tid not in _w_target_parsed_cache:
        c2, norm_n2, norm_a2 = _w_target_raw[tid]
        toks_n2 = set(norm_n2.split())
        toks_a2 = set(norm_a2.split())
        digits_a2 = extract_digits_set(norm_a2)
        _w_target_parsed_cache[tid] = (
            c2, norm_n2, norm_a2,
            toks_n2, toks_a2, digits_a2,
            len(norm_n2), len(norm_a2),
            len(toks_n2), len(toks_a2),
            len(digits_a2)
        )
    return _w_target_parsed_cache[tid]

def _worker_process_batch(tasks):
    """
    tasks: list of (s1_id, r_n1, r_a1, c1, candidate_ids_str)
    Returns: list of (s1_id, matched_str), candidates_evaluated_count, matches_count
    """
    results = []
    cands_eval = 0
    matches_cnt = 0

    for s1_id, r_n1, r_a1, c1, cand_str in tasks:
        if not cand_str:
            results.append(f"{s1_id}\t\n")
            continue

        cands = cand_str.split(",")
        cands_eval += len(cands)

        norm_n1 = fast_norm_name(r_n1)
        norm_a1 = fast_norm_addr(r_a1)
        c1 = c1 if c1 != "UNKNOWN" else ""
        toks_n1 = set(norm_n1.split())
        toks_a1 = set(norm_a1.split())
        digits_a1 = extract_digits_set(norm_a1)
        len_n1 = len(norm_n1)
        len_a1 = len(norm_a1)
        len_tn1 = len(toks_n1)
        len_ta1 = len(toks_a1)
        len_d1 = len(digits_a1)

        best_cand = None
        best_score = -1.0

        for tid in cands:
            if tid not in _w_target_raw:
                continue

            t = _get_parsed_target(tid)
            c2, norm_n2, norm_a2, toks_n2, toks_a2, digits_a2 = t[0], t[1], t[2], t[3], t[4], t[5]
            len_n2, len_a2, len_tn2, len_ta2, len_d2 = t[6], t[7], t[8], t[9], t[10]

            score = 0.0
            n_exact = (norm_n1 == norm_n2 and len_n1 >= 3)
            if n_exact:
                score += 55.0
            else:
                if len_tn1 > 0 and len_tn2 > 0:
                    n_ov = len(toks_n1 & toks_n2)
                    n_jac = n_ov / (len_tn1 + len_tn2 - n_ov)
                    score += n_jac * 45.0

            if len_ta1 > 0 and len_ta2 > 0:
                a_ov = len(toks_a1 & toks_a2)
                a_jac = a_ov / (len_ta1 + len_ta2 - a_ov)
                score += a_jac * 25.0

            if len_d1 > 0 and len_d2 > 0:
                d_ov = len(digits_a1 & digits_a2)
                d_jac = d_ov / (len_d1 + len_d2 - d_ov)
                score += d_jac * 15.0

            if c1 and c2 and c1 == c2:
                score += 5.0

            if len_n1 >= 4 and len_n2 >= 4 and norm_n1[:4] == norm_n2[:4]:
                score += 5.0

            len_diff_n = abs(len_n1 - len_n2)
            if len_diff_n > 10:
                score -= min(15.0, (len_diff_n - 10) * 0.5)

            if score > best_score:
                best_score = score
                best_cand = tid

        # Acceptance threshold
        if best_cand and best_score >= 48.0:
            results.append(f"{s1_id}\t{best_cand}\n")
            matches_cnt += 1
        else:
            results.append(f"{s1_id}\t\n")

    return results, cands_eval, matches_cnt

def get_ram_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def get_cpu_pct():
    return psutil.cpu_percent(interval=None)

def main():
    t0_start = time.time()
    print("=" * 80, flush=True)
    print("=== FAST DETERMINISTIC ENTITY MATCHING PIPELINE ===", flush=True)
    print("=" * 80, flush=True)

    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    if not cand_file.exists():
        print("ERROR: output/candidate_pairs.tsv does not exist!", flush=True)
        sys.exit(1)

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Launching worker pool with {num_workers} processes...", flush=True)

    pool = mp.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(str(s2_path), str(s3_path))
    )
    print(f"Worker pool initialized across {num_workers} workers in {time.time() - t0_start:.2f}s.\n", flush=True)

    total_s1_records = 1732544
    batch_size = 5000  # Process in 5,000 S1 batches
    subchunk_size = 1000

    s1_stream = open(s1_path, "r", encoding="utf-8")
    cand_stream = open(cand_file, "r", encoding="utf-8")

    # Skip headers
    s1_stream.readline()
    cand_stream.readline()

    out_file = open(match_file, "w", encoding="utf-8")
    out_file.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_processed = 0
    total_cands_processed = 0
    total_matches_produced = 0

    t0_scoring = time.time()
    last_log_time = time.time()

    while total_s1_processed < total_s1_records:
        batch_tasks = []
        for _ in range(batch_size):
            s1_line = s1_stream.readline()
            cand_line = cand_stream.readline()
            if not s1_line:
                break
            
            s1_parts = s1_line.rstrip("\n").split("\t")
            cand_parts = cand_line.rstrip("\n").split("\t")

            s1_id = s1_parts[0]
            r_n1 = s1_parts[1] if len(s1_parts) > 1 else ""
            r_a1 = s1_parts[2] if len(s1_parts) > 2 else ""
            c1 = s1_parts[3] if len(s1_parts) > 3 else ""

            cand_str = cand_parts[1] if len(cand_parts) > 1 else ""
            batch_tasks.append((s1_id, r_n1, r_a1, c1, cand_str))

        if not batch_tasks:
            break

        # Split batch into subchunk tasks for workers
        subchunks = [batch_tasks[i:i + subchunk_size] for i in range(0, len(batch_tasks), subchunk_size)]
        worker_results = pool.map(_worker_process_batch, subchunks)

        for res_lines, c_eval, m_cnt in worker_results:
            out_file.writelines(res_lines)
            total_s1_processed += len(res_lines)
            total_cands_processed += c_eval
            total_matches_produced += m_cnt

        out_file.flush()

        # LIVE PROGRESS LOGGING (every 5-10s or every batch)
        now = time.time()
        if now - last_log_time >= 5.0 or total_s1_processed >= total_s1_records:
            last_log_time = now
            elapsed = now - t0_scoring
            rate = total_s1_processed / elapsed if elapsed > 0 else 0
            rem_s1 = total_s1_records - total_s1_processed
            eta = rem_s1 / rate if rate > 0 else 0
            pct = (total_s1_processed / total_s1_records) * 100.0

            cpu = get_cpu_pct()
            ram = get_ram_gb()

            log_line = (
                f"S1 processed: {total_s1_processed:,} / {total_s1_records:,} | "
                f"Progress: {pct:.2f}% | "
                f"Matches: {total_matches_produced:,} | "
                f"Candidates: {total_cands_processed:,} | "
                f"Rate: {rate:.1f} S1/sec | "
                f"ETA: {format_time(eta)} | "
                f"CPU: {cpu:.1f}% | "
                f"RAM: {ram:.2f} GB | "
                f"Status: ACTIVE STREAMING"
            )
            print(log_line, flush=True)

    s1_stream.close()
    cand_stream.close()
    out_file.close()
    pool.close()
    pool.join()

    total_time = time.time() - t0_start
    print("\n" + "=" * 80, flush=True)
    print(f"MATCHING COMPLETE! Total time: {format_time(total_time)}", flush=True)
    print(f"Total S1: {total_s1_processed:,} | Total Matches: {total_matches_produced:,} | File: {match_file}", flush=True)
    print("=" * 80, flush=True)

    # --- AUTOMATIC SUBMISSION VALIDATION ---
    print("\nExecuting Official Validator script...", flush=True)
    val_script = base_dir / "utils" / "validate_submission.py"
    val_cmd = f"py -3 \"{val_script}\" --matching \"{match_file}\" --candidate \"{cand_file}\" --test-dir \"{test_dir}\""
    print(f"Running: {val_cmd}\n", flush=True)

    os.system(val_cmd)

if __name__ == "__main__":
    main()
