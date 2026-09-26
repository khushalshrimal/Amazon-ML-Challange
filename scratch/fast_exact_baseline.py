"""
Ultra-Fast Cheap Exact Deterministic Entity Matching & Official Submission Generator.
Processes all 1,732,544 S1 records with 5-second startup and 0 fuzzy overhead.
Prints live progress every 5 seconds and automatically executes official validation.
"""

import os
import sys
import time
import string
import psutil
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from progress_logger import format_time

trans_table = str.maketrans({c: ' ' for c in string.punctuation})

def ultra_fast_norm(t):
    if not isinstance(t, str) or not t:
        return ""
    return " ".join(t.lower().replace("&", " and ").translate(trans_table).split())

def get_ram_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def get_cpu_pct():
    return psutil.cpu_percent(interval=None)

def main():
    t0_start = time.time()
    print("=" * 80, flush=True)
    print("=== ULTRA-FAST CHEAP EXACT MATCHING SUBMISSION PIPELINE ===", flush=True)
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

    # --- STEP 1: FAST RAW TARGET INDEXING (5 SECONDS) ---
    print("\n--- STEP 1: Fast Raw Target Indexing (Source-2 & Source-3) ---", flush=True)
    t0_target = time.time()
    
    target_raw = {}
    target_norm_cache = {}

    for fpath in [s2_path, s3_path]:
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    target_raw[parts[0]] = (parts[1], parts[2])

    print(f"Target Store Loaded: {len(target_raw):,} records in {time.time() - t0_target:.2f}s | RAM: {get_ram_gb():.2f} GB\n", flush=True)

    def get_target_norm(tid):
        if tid not in target_norm_cache:
            r_n, r_a = target_raw[tid]
            target_norm_cache[tid] = (ultra_fast_norm(r_n), ultra_fast_norm(r_a))
        return target_norm_cache[tid]

    # --- STEP 2: STREAM S1 & CANDIDATE PAIRS AND MATCH ---
    print("--- STEP 2: Streaming Cheap Exact Matching ---", flush=True)
    total_s1_records = 1732544
    s1_stream = open(s1_path, "r", encoding="utf-8")
    cand_stream = open(cand_file, "r", encoding="utf-8")

    # Skip headers
    s1_stream.readline()
    cand_stream.readline()

    out_file = open(match_file, "w", encoding="utf-8")
    out_file.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_processed = 0
    total_matches_produced = 0

    t0_scoring = time.time()
    last_log_time = time.time()

    write_buffer = []
    buffer_size = 50000

    for s1_line in s1_stream:
        cand_line = cand_stream.readline()
        if not cand_line:
            break

        s1_parts = s1_line.rstrip("\n").split("\t")
        cand_parts = cand_line.rstrip("\n").split("\t")

        s1_id = s1_parts[0]
        r_n1 = s1_parts[1] if len(s1_parts) > 1 else ""
        r_a1 = s1_parts[2] if len(s1_parts) > 2 else ""

        cand_str = cand_parts[1] if len(cand_parts) > 1 else ""
        cands = cand_str.split(",") if cand_str else []

        if not cands:
            write_buffer.append(f"{s1_id}\t\n")
            total_s1_processed += 1
        else:
            norm_n1 = ultra_fast_norm(r_n1)
            norm_a1 = ultra_fast_norm(r_a1)
            len_n1 = len(norm_n1)

            matched_id = ""

            for tid in cands:
                if tid not in target_raw:
                    continue

                t_n2, t_a2 = get_target_norm(tid)

                # Rule 1: Exact Name match (length >= 3)
                if len_n1 >= 3 and norm_n1 == t_n2:
                    matched_id = tid
                    break

                # Rule 2: Exact Name prefix 8 chars + Exact Address match
                if len_n1 >= 8 and len(t_n2) >= 8 and norm_n1[:8] == t_n2[:8] and norm_a1 == t_a2 and norm_a1 != "":
                    matched_id = tid
                    break

            if matched_id:
                write_buffer.append(f"{s1_id}\t{matched_id}\n")
                total_matches_produced += 1
            else:
                write_buffer.append(f"{s1_id}\t\n")

            total_s1_processed += 1

        if len(write_buffer) >= buffer_size:
            out_file.writelines(write_buffer)
            out_file.flush()
            write_buffer.clear()

        # LIVE PROGRESS LOGGING (every 5 seconds)
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
                f"Rate: {rate:,.1f} S1/sec | "
                f"ETA: {format_time(eta)} | "
                f"CPU: {cpu:.1f}% | "
                f"RAM: {ram:.2f} GB"
            )
            print(log_line, flush=True)

    if write_buffer:
        out_file.writelines(write_buffer)
        out_file.flush()
        write_buffer.clear()

    s1_stream.close()
    cand_stream.close()
    out_file.close()

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
