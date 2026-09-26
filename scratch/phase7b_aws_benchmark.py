import os
import sys
import time
import re
import gc
import json
import psutil
import joblib
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

base_dir = Path(r"c:\Users\khush\OneDrive\Desktop\amzon ml challenge")
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
sys.path.append(str(src_dir))

from blocking import EntityBlocker

# GLOBAL WORKER READ-ONLY DATA STRUCTURES
global_blocker = None
global_target_attr = None
global_model = None
target_cache = {}

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')

def fast_norm_name(text):
    if not isinstance(text, str) or pd.isna(text): return ""
    return sub_space.sub(' ', sub_punct.sub(' ', sub_amp.sub(' and ', text.lower()))).strip()

def fast_norm_addr(text):
    if not isinstance(text, str) or pd.isna(text): return ""
    return sub_space.sub(' ', sub_punct.sub(' ', text.lower())).strip()

def extract_digits(text: str) -> set:
    return set(re.findall(r'\b\d+\b', text))

def extract_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n: return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# WORKER INITIALIZER (Loads pre-built index directly inside worker from disk cache)
def init_worker(blocker_cache_path_str, model_path_str):
    global global_blocker, global_target_attr, global_model
    global_blocker, global_target_attr = joblib.load(blocker_cache_path_str)
    global_model = joblib.load(model_path_str)

def get_target_parsed(tid):
    if tid not in target_cache:
        rec_str = global_target_attr[tid]
        c2, norm_n2, norm_a2 = rec_str.split('\t')
        target_cache[tid] = (
            c2, norm_n2, norm_a2,
            set(norm_n2.split()),
            set(norm_a2.split()),
            extract_ngrams(norm_n2, 3),
            extract_ngrams(norm_a2, 3),
            extract_digits(norm_a2)
        )
    return target_cache[tid]

def extract_fast_features_worker(a1, tid):
    c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = get_target_parsed(tid)
    
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
    is_s2 = 1.0 if tid.startswith('S2-') else 0.0

    return [
        n_exact, n_jaccard, n_char_jaccard, a_jaccard, a_char_jaccard,
        d_jaccard, c_match, len_diff_n, len_diff_a, float(n_overlap),
        float(a_overlap), missing_addr, is_s2
    ]

# WORKER PROCESS FUNCTION
def process_s1_chunk(s1_records_chunk):
    cand_lines = []
    match_lines = []
    
    total_candidates = 0
    total_matches = 0
    zero_matches = 0

    for s1_id, r_n, r_a, c in s1_records_chunk:
        norm_n = fast_norm_name(r_n)
        norm_a = fast_norm_addr(r_a)
        c_str = str(c)
        
        toks_n = set(norm_n.split())
        toks_a = set(norm_a.split())
        ngs_n = extract_ngrams(norm_n, 3)
        ngs_a = extract_ngrams(norm_a, 3)
        digits_a = extract_digits(norm_a)

        # Fast Candidate Retrieval using Strategy 3 Blocker Index
        cands = set()
        n_toks = list(toks_n)
        a_toks = list(toks_a)
        ngs = list(ngs_n)

        if len(n_toks) >= 2:
            cands.update(global_blocker.name_pref2_idx.get((" ".join(n_toks[:2]), c_str), set()))
        elif n_toks:
            cands.update(global_blocker.name_pref2_idx.get((n_toks[0], c_str), set()))

        if len(a_toks) >= 2:
            cands.update(global_blocker.addr_pref2_idx.get((" ".join(a_toks[:2]), c_str), set()))
        elif a_toks:
            cands.update(global_blocker.addr_pref2_idx.get((a_toks[0], c_str), set()))

        for tok in [t for t in n_toks if t not in global_blocker.stop_name_tokens and len(t) > 2][:2]:
            cands.update(global_blocker.name_rare_tok_idx.get((tok, c_str), set()))

        for tok in [t for t in a_toks if t not in global_blocker.stop_addr_tokens and (len(t) > 2 or t.isdigit())][:2]:
            cands.update(global_blocker.addr_rare_tok_idx.get((tok, c_str), set()))

        for ng in [g for g in ngs if g not in global_blocker.stop_ngrams][:1]:
            cands.update(global_blocker.name_ngram_idx.get((ng, c_str), set()))

        cand_list = list(cands)
        total_candidates += len(cand_list)
        cand_lines.append(f"{s1_id}\t{','.join(cand_list)}\n")

        # Feature Extraction & Model Scoring if Candidates Exist
        valid_cands = [tid for tid in cand_list if tid in global_target_attr]
        matched_tids = []

        if valid_cands:
            a1_dict = {
                'norm_n': norm_n, 'toks_n': toks_n, 'ngrams_n': ngs_n,
                'norm_a': norm_a, 'toks_a': toks_a, 'ngrams_a': ngs_a,
                'digits_a': digits_a, 'c': c_str
            }
            
            X_mat = np.zeros((len(valid_cands), 13), dtype=np.float32)
            for idx, tid in enumerate(valid_cands):
                X_mat[idx] = extract_fast_features_worker(a1_dict, tid)

            probs = global_model.predict_proba(X_mat)[:, 1]
            for tid, prob in zip(valid_cands, probs):
                if prob >= 0.98:
                    matched_tids.append(tid)

        if matched_tids:
            total_matches += len(matched_tids)
            match_lines.append(f"{s1_id}\t{','.join(matched_tids)}\n")
        else:
            zero_matches += 1
            match_lines.append(f"{s1_id}\t\n")

    return cand_lines, match_lines, total_candidates, total_matches, zero_matches

def main():
    parser = argparse.ArgumentParser(description="Parallel AWS 10k Benchmark")
    parser.add_argument("--max-s1", type=int, default=10000, help="Number of S1 rows to benchmark (default: 10000)")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of worker processes (default: 6)")
    args = parser.parse_args()

    print("=== AWS DISTRIBUTED/PARALLEL INFERENCE PIPELINE BENCHMARK ===", flush=True)
    print(f"Target S1 Benchmark Size: {args.max_s1:,} rows", flush=True)
    print(f"Worker Processes: {args.num_workers} processes", flush=True)
    print(f"Initial Host RAM: {get_ram_mb():.1f} MB", flush=True)

    t0 = time.time()
    
    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"
    blocker_cache_path = base_dir / "models" / "blocker_index.joblib"
    model_path = base_dir / "models" / "lightgbm_model.joblib"

    # STAGE 1: LOAD OR BUILD TARGET INDEX & TARGET ATTR DICT
    print("\n--- STAGE 1: Target Pool Load & Strategy 3 Index Initialization ---", flush=True)
    t_idx_start = time.time()

    blocker = EntityBlocker()
    target_attr = {}

    if blocker_cache_path.exists():
        print(f"Loading pre-built Blocker index from {blocker_cache_path}...", flush=True)
        blocker, target_attr = joblib.load(blocker_cache_path)
        t_index_build = time.time() - t_idx_start
        print(f"Loaded target index in {t_index_build:.2f}s! Target count: {len(target_attr):,}. Host RAM: {get_ram_mb():.1f} MB", flush=True)
    else:
        print("Building target index from Source-2 and Source-3 files...", flush=True)
        s2_path = test_dir / "test_source2.tsv"
        s3_path = test_dir / "test_source3.tsv"

        def add_target_file(file_path, label):
            count = 0
            for chunk in pd.read_csv(file_path, sep="\t", chunksize=100000, dtype=str):
                eids = chunk['entity_id'].values
                r_names = chunk['business_name'].values
                r_addrs = chunk['business_address'].values
                countries = chunk['country'].fillna('UNKNOWN').astype(str).values

                for mid, rn, ra, c in zip(eids, r_names, r_addrs, countries):
                    nn = fast_norm_name(rn)
                    na = fast_norm_addr(ra)
                    target_attr[mid] = f"{c}\t{nn}\t{na}"
                    blocker.add_target_record(mid, str(c), nn, na)
                    count += 1
            print(f"Loaded {label}: {count:,} records into target index.", flush=True)

        add_target_file(s2_path, "Source-2")
        add_target_file(s3_path, "Source-3")

        t_index_build = time.time() - t_idx_start
        print(f"Index build complete in {t_index_build:.2f}s! Total target pool: {len(target_attr):,}. Saving to disk cache...", flush=True)
        joblib.dump((blocker, target_attr), blocker_cache_path, compress=1)
        print("Blocker index saved to disk cache.", flush=True)

    blocker_file_size_mb = os.path.getsize(blocker_cache_path) / (1024 * 1024)

    # STAGE 2: READ S1 BENCHMARK DATASET
    print(f"\n--- STAGE 2: Reading {args.max_s1:,} Test S1 Entities ---", flush=True)
    s1_path = test_dir / "test_source1.tsv"
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=args.max_s1, dtype=str)
    
    s1_tuples = list(zip(
        df_s1['entity_id'].values,
        df_s1['business_name'].values,
        df_s1['business_address'].values,
        df_s1['country'].fillna('UNKNOWN').astype(str).values
    ))
    
    print(f"Loaded {len(s1_tuples):,} S1 benchmark records. Partitioning into worker batches...", flush=True)

    # Chunk S1 tuples into worker tasks (500 per task)
    batch_size = 500
    s1_chunks = [s1_tuples[i:i + batch_size] for i in range(0, len(s1_tuples), batch_size)]

    # STAGE 3: MULTIPROCESSING PARALLEL EXECUTION (Pass file path strings in initargs to avoid Windows IPC MemoryError)
    print(f"\n--- STAGE 3: Running Parallel Benchmark ({args.num_workers} Workers across {len(s1_chunks)} Batches) ---", flush=True)
    
    t_parallel_start = time.time()
    
    all_cand_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
    all_match_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    
    total_candidates = 0
    total_matches = 0
    total_zero_s1 = 0

    with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(str(blocker_cache_path), str(model_path))) as pool:
        results = pool.map(process_s1_chunk, s1_chunks)

    t_parallel = time.time() - t_parallel_start
    peak_ram = get_ram_mb()

    for c_lines, m_lines, n_cands, n_matches, n_zeros in results:
        all_cand_lines.extend(c_lines)
        all_match_lines.extend(m_lines)
        total_candidates += n_cands
        total_matches += n_matches
        total_zero_s1 += n_zeros

    # STAGE 4: WRITE BENCHMARK OUTPUT FILES & VALIDATE
    print("\n--- STAGE 4: Writing Output Files & Running Validator ---", flush=True)
    with open(cand_file, "w", encoding="utf-8") as f:
        f.writelines(all_cand_lines)
    with open(match_file, "w", encoding="utf-8") as f:
        f.writelines(all_match_lines)

    print(f"Wrote {len(all_cand_lines)-1:,} candidate rows to {cand_file}", flush=True)
    print(f"Wrote {len(all_match_lines)-1:,} matching rows to {match_file}", flush=True)

    # Run official validator
    import subprocess
    val_script = base_dir / "utils" / "validate_submission.py"
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(test_dir)
    ]
    val_res = subprocess.run(cmd, capture_output=True, text=True)
    val_out = val_res.stdout.strip() + "\n" + val_res.stderr.strip()

    # METRICS CALCULATION
    s1_processed = len(s1_tuples)
    s1_per_sec = s1_processed / t_parallel if t_parallel > 0 else 0
    cands_per_sec = total_candidates / t_parallel if t_parallel > 0 else 0
    
    full_s1_total = 1732544
    est_full_runtime_s = full_s1_total / s1_per_sec if s1_per_sec > 0 else 0
    est_full_runtime_min = est_full_runtime_s / 60.0

    print("\n=======================================================", flush=True)
    print("           10,000-S1 BENCHMARK PERFORMANCE REPORT       ", flush=True)
    print("=======================================================", flush=True)
    print(f"1. Target Index Load Time:    {t_index_build:.2f} seconds")
    print(f"2. blocker_index.joblib Size: {blocker_file_size_mb:.2f} MB")
    print(f"3. Peak Host RAM Usage:      {peak_ram:.1f} MB ({peak_ram/1024:.2f} GB)")
    print(f"4. CPU Utilization:         {args.num_workers} Worker Processes active")
    print(f"5. 10,000-S1 Benchmark Runtime: {t_parallel:.2f} seconds")
    print(f"6. Candidate Pairs Generated: {total_candidates:,}")
    print(f"7. Candidate Pairs / Sec:     {cands_per_sec:,.2f} / sec")
    print(f"8. S1 Rows / Sec:             {s1_per_sec:.2f} S1 / sec")
    print(f"9. Projected Full 1.73M S1 Runtime (6 Workers): {est_full_runtime_min:.1f} minutes ({est_full_runtime_min/60:.2f} hours)")
    print("10. Validation Check Result:\n" + val_out)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
