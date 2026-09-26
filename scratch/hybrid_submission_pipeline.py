"""
Hybrid 2-PASS Memory-Safe C++ / Python Submission Pipeline for Amazon ML Challenge.
Combines 2-PASS memory-safe C++ candidate generator (Peak RAM < 400 MB) with Python LightGBM ML scoring.
Guarantees live progress logging every <= 2-5 seconds and official validation.
"""

import os
import sys
import time
import subprocess
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(str(src_dir)) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from inference_optimized import OnDemandTargetStore, fast_norm_name, fast_norm_addr, extract_digits, extract_ngrams
from progress_logger import ProgressLogger, format_time

def main():
    t0_pipeline = time.time()
    logger = ProgressLogger(min_interval_sec=2.0, pct_step=2.0)
    
    print("=" * 80, flush=True)
    print("=== HYBRID 2-PASS C++ / PYTHON SUBMISSION PIPELINE ===", flush=True)
    print("=" * 80, flush=True)

    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"
    model_path = base_dir / "models" / "lightgbm_model.joblib"
    cpp_exe = base_dir / "code" / "cpp_blocker_memory_safe.exe"
    cpp_src = base_dir / "code" / "cpp_blocker_memory_safe.cpp"

    # STAGE 1: COMPILE & RUN 2-PASS C++ CANDIDATE GENERATOR
    logger.start_stage(1, 3, "Running 2-PASS C++ Fast Target Indexing & Candidate Generation (Posting Cap: 100)")
    
    if not cpp_exe.exists() or cpp_src.stat().st_mtime > cpp_exe.stat().st_mtime:
        print("Compiling code/cpp_blocker_memory_safe.cpp with g++ -O3...", flush=True)
        compile_cmd = ["g++", "-O3", "-std=c++14", str(cpp_src), "-o", str(cpp_exe), "-lpsapi"]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("C++ Compilation Error:\n", res.stderr, flush=True)
            sys.exit(1)
        print("Compiled C++ executable successfully.", flush=True)

    t0_cpp = time.time()
    cpp_proc = subprocess.Popen([str(cpp_exe), str(base_dir), "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Read live C++ terminal output
    for line in iter(cpp_proc.stdout.readline, ''):
        line_clean = line.strip()
        if line_clean:
            print(line_clean, flush=True)
            
    cpp_proc.stdout.close()
    return_code = cpp_proc.wait()
    
    if return_code != 0:
        print(f"ERROR: C++ candidate generator failed with code {return_code}", flush=True)
        sys.exit(1)

    t_cpp = time.time() - t0_cpp
    logger.finish_stage("C++ Candidate Generation")

    # STAGE 2: PYTHON TARGET STORE PREPARATION FOR ML FEATURE EXTRACTION
    logger.start_stage(2, 3, "Building Target Attribute Cache for Python Feature Extraction")
    t0_target = time.time()
    
    target_store = OnDemandTargetStore()
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    s1_path = test_dir / "test_source1.tsv"

    total_target_loaded = 0
    total_expected_targets = 9969589

    for fpath, label in [(s2_path, "Source-2"), (s3_path, "Source-3")]:
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
            eids = chunk['entity_id'].values
            raw_names = chunk['business_name'].values
            raw_addrs = chunk['business_address'].values
            countries = chunk['country'].fillna('UNKNOWN').astype(str).values
            
            for mid, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
                norm_n = fast_norm_name(r_n)
                norm_a = fast_norm_addr(r_a)
                target_store.add_target_raw(mid, c, norm_n, norm_a)
                total_target_loaded += 1
            
            logger.log_progress(total_target_loaded, total_expected_targets)

    logger.finish_stage("Target Attribute Cache Build")

    # STAGE 3: LIGHTGBM MODEL SCORING & MATCHING_RESULTS.TSV GENERATION
    logger.start_stage(3, 3, "Running LightGBM Feature Extraction & Scoring (Threshold: 0.98)")
    t0_score = time.time()

    model = joblib.load(model_path)
    print(f"Loaded LightGBM model from {model_path.name}", flush=True)

    # Read S1 test entities to preserve exact query order
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    s1_queries = []
    for _, r in s1_df.iterrows():
        s1_id = r['entity_id']
        c = r['country'] if pd.notna(r['country']) else "UNKNOWN"
        r_n = r['business_name'] if pd.notna(r['business_name']) else ""
        r_a = r['business_address'] if pd.notna(r['business_address']) else ""
        
        norm_n = fast_norm_name(r_n)
        norm_a = fast_norm_addr(r_a)
        
        s1_queries.append({
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

    # Read candidates from generated candidate_pairs.tsv
    cand_map = {}
    with open(cand_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            parts = line_str.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                cand_map[s1_id] = cand_str.split(",")
            else:
                cand_map[s1_id] = []

    # Write matching_results.tsv
    with open(match_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1 = len(s1_queries)
    total_candidates_evaluated = 0
    total_matches_found = 0

    batch_size = 2000
    for i in range(0, total_s1, batch_size):
        batch = s1_queries[i:i + batch_size]
        
        pair_meta = []
        for s1_rec in batch:
            s1_id = s1_rec['s1_id']
            cands = cand_map.get(s1_id, [])
            total_candidates_evaluated += len(cands)
            for tid in cands:
                if tid in target_store.raw_attr:
                    pair_meta.append((s1_rec, tid))

        match_lines_dict = defaultdict(list)

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

        batch_out_lines = []
        for s1_rec in batch:
            s1_id = s1_rec['s1_id']
            m_list = match_lines_dict.get(s1_id, [])
            if m_list:
                total_matches_found += len(m_list)
                match_str = ",".join(m_list)
            else:
                match_str = ""
            batch_out_lines.append(f"{s1_id}\t{match_str}\n")

        with open(match_file, "a", encoding="utf-8") as f_out:
            f_out.writelines(batch_out_lines)

        current_processed = min(i + batch_size, total_s1)
        logger.log_progress(
            current=current_processed,
            total=total_s1,
            candidates=total_evaluated_cands,
            matches=total_matches_found
        )

    t_score = time.time() - t0_score
    logger.finish_stage("LightGBM Model Scoring")

    # STAGE 5: VERIFY OUTPUTS & RUN OFFICIAL SUBMISSION VALIDATOR
    logger.start_stage(5, 5, "Verifying Submission Outputs & Running Official Validator")
    
    t_total = time.time() - t0_pipeline
    print("\n" + "=" * 80, flush=True)
    print("=== FINAL SUBMISSION PIPELINE SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    print(f"1. Total Pipeline Runtime:      {t_total:.2f} s ({t_total/60.0:.2f} min)", flush=True)
    print(f"2. 2-Pass C++ Candidate Time:   {t_cpp:.2f} s", flush=True)
    print(f"3. Python Model Scoring Time:   {t_score:.2f} s", flush=True)
    print(f"4. Total S1 Records Processed: {total_s1:,}", flush=True)
    print(f"5. Total Candidates Evaluated: {total_evaluated_cands:,}", flush=True)
    print(f"6. Total Matches Found (T>=0.98): {total_matches_found:,}", flush=True)
    
    cand_bytes = cand_file.stat().st_size if cand_file.exists() else 0
    match_bytes = match_file.stat().st_size if match_file.exists() else 0
    print(f"7. output/candidate_pairs.tsv:  {cand_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"8. output/matching_results.tsv: {match_bytes / (1024*1024):.2f} MB", flush=True)
    print("=" * 80 + "\n", flush=True)

    validator_script = base_dir / "utils" / "validate_submission.py"
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(test_dir)
    ]

    print("Running Official Submission Validator:", " ".join(cmd), flush=True)
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
