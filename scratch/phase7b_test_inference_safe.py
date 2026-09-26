"""
Resumable & Multiprocess Full Submission Generator for Amazon ML Challenge.
Target Pool: 9,969,589 records (S2 + S3)
Query Pool: 1,732,544 records (S1)
Posting-List Pruning Cap: 100
Batch Size: 1,000
Model: LightGBM (T=0.98, 13 features)
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

from blocking import EntityBlocker, extract_3grams
from progress_logger import ProgressLogger, format_time

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')

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

def extract_digits(text: str) -> set:
    return set(re.findall(r'\b\d+\b', text))

def extract_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

# Global worker variables for process pool
_g_target_attr = None
_g_blocker = None
_g_target_cache = {}

def _init_worker(target_attr_dict, blocker_obj):
    global _g_target_attr, _g_blocker, _g_target_cache
    _g_target_attr = target_attr_dict
    _g_blocker = blocker_obj
    _g_target_cache = {}

def _get_target_parsed(mid2):
    if mid2 not in _g_target_cache:
        rec_str = _g_target_attr[mid2]
        c2, norm_n2, norm_a2 = rec_str.split('\t')
        _g_target_cache[mid2] = (
            c2, norm_n2, norm_a2,
            set(norm_n2.split()),
            set(norm_a2.split()),
            extract_ngrams(norm_n2, 3),
            extract_ngrams(norm_a2, 3),
            extract_digits(norm_a2)
        )
    return _g_target_cache[mid2]

def extract_fast_features_worker(a1, mid2):
    c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = _get_target_parsed(mid2)
    
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
        n_exact,
        n_jaccard,
        n_char_jaccard,
        a_jaccard,
        a_char_jaccard,
        d_jaccard,
        c_match,
        len_diff_n,
        len_diff_a,
        float(n_overlap),
        float(a_overlap),
        missing_addr,
        is_s2
    ]

def _process_subchunk(subchunk_rows):
    """
    Worker function to process a list of S1 rows (tuples: s1_id, r_n, r_a, c).
    Returns list of tuples: (s1_id, cand_str, valid_tids, feature_rows)
    """
    results = []
    for s1_id, r_n, r_a, c in subchunk_rows:
        norm_n = fast_norm_name(r_n)
        norm_a = fast_norm_addr(r_a)
        
        a1 = {
            'norm_n': norm_n,
            'toks_n': set(norm_n.split()),
            'ngrams_n': extract_ngrams(norm_n, 3),
            'norm_a': norm_a,
            'toks_a': set(norm_a.split()),
            'ngrams_a': extract_ngrams(norm_a, 3),
            'digits_a': extract_digits(norm_a),
            'c': c
        }
        
        raw_rec = {
            'entity_id': s1_id,
            'business_name': norm_n,
            'business_address': norm_a,
            'country': c
        }
        
        cands = _g_blocker.get_candidates(raw_rec)
        cand_str = ",".join(cands)
        
        valid_tids = [tid for tid in cands if tid in _g_target_attr]
        
        if valid_tids:
            feats = [extract_fast_features_worker(a1, tid) for tid in valid_tids]
        else:
            feats = []
            
        results.append((s1_id, cand_str, valid_tids, feats))
    return results

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    logger = ProgressLogger(min_interval_sec=60.0, pct_step=5.0)
    
    print("=" * 80, flush=True)
    print("=== RESUMABLE & MULTIPROCESS FULL SUBMISSION GENERATOR ===", flush=True)
    print("=" * 80, flush=True)
    print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    cand_file = output_dir / "candidate_pairs.tsv"
    match_file = output_dir / "matching_results.tsv"
    checkpoint_file = output_dir / "checkpoint.json"
    blocker_cache_path = base_dir / "models" / "blocker_index_pruned100.joblib"
    model_path = base_dir / "models" / "lightgbm_model.joblib"

    # STAGE 1: LOAD MODEL & BUILD TARGET POOL INDEX
    logger.start_stage(1, 4, "Building Target Index & Loading Blocker (Posting Limit: 100)")
    t0_start = time.time()

    model = joblib.load(model_path)
    print(f"Loaded LightGBM model from {model_path.name}", flush=True)

    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    s1_path = test_dir / "test_source1.tsv"

    target_attr = {}
    blocker = EntityBlocker()

    if blocker_cache_path.exists():
        print(f"Loading pre-built 100-pruned Blocker index from disk cache ({blocker_cache_path.name})...", flush=True)
        blocker, target_attr = joblib.load(blocker_cache_path)
        print(f"Loaded Blocker index from disk cache across {len(target_attr):,} target records.", flush=True)
    else:
        total_target_loaded = 0
        total_target_records = 9969589
        target_records = []
        
        for fpath, label in [(s2_path, "Source-2"), (s3_path, "Source-3")]:
            print(f"Loading {label} ({fpath.name})...", flush=True)
            for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
                eids = chunk['entity_id'].values
                raw_names = chunk['business_name'].values
                raw_addrs = chunk['business_address'].values
                countries = chunk['country'].fillna('UNKNOWN').astype(str).values
                
                for mid, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
                    norm_n = fast_norm_name(r_n)
                    norm_a = fast_norm_addr(r_a)
                    target_attr[mid] = f"{c}\t{norm_n}\t{norm_a}"
                    target_records.append({
                        'entity_id': mid,
                        'country': c,
                        'business_name': norm_n,
                        'business_address': norm_a
                    })
                    total_target_loaded += 1
                    
                logger.log_progress(total_target_loaded, total_target_records)

        print("Building Blocker inverted indexes...", flush=True)
        blocker.build_index(target_records)
        del target_records
        gc.collect()

        # Apply posting-list limit = 100
        print("Applying posting-list cap = 100 across blocker channels...", flush=True)
        MAX_LIMIT = 100
        pruned_cnt = 0
        for ch in [blocker.name_pref2_idx, blocker.addr_pref2_idx, blocker.name_rare_tok_idx, blocker.addr_rare_tok_idx, blocker.name_ngram_idx]:
            to_remove = [k for k, s in ch.items() if len(s) > MAX_LIMIT]
            for k in to_remove:
                del ch[k]
                pruned_cnt += 1
        print(f"Pruned {pruned_cnt:,} generic keys exceeding limit = 100.", flush=True)

        print(f"Saving 100-pruned Blocker index to disk cache ({blocker_cache_path.name})...", flush=True)
        joblib.dump((blocker, target_attr), blocker_cache_path, compress=1)
        print("Blocker index saved to disk cache.", flush=True)

    t_index_build = time.time() - t0_start
    logger.finish_stage("Target Index & Blocker Build")

    # STAGE 2: CHECKPOINT & OUTPUT FILE INITIALIZATION
    logger.start_stage(2, 4, "Initializing Output Files & Resumable Checkpoint State")
    completed_batch_idx = 0
    total_s1_entities = 0
    total_candidate_pairs = 0
    total_predicted_matches = 0
    total_zero_match_s1 = 0

    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
            completed_batch_idx = ckpt_data.get("completed_batches", 0)
            total_s1_entities = ckpt_data.get("processed_s1", 0)
            total_candidate_pairs = ckpt_data.get("processed_candidates", 0)
            total_predicted_matches = ckpt_data.get("predicted_matches", 0)
            total_zero_match_s1 = ckpt_data.get("zero_match_s1", 0)
        print(f"RESUMING FROM CHECKPOINT! Already completed: {completed_batch_idx} batches ({total_s1_entities:,} S1 entities).", flush=True)
    else:
        print("Starting fresh inference run. Initializing output headers...", flush=True)
        with open(cand_file, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(match_file, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    logger.finish_stage("Checkpoint Initialization")

    # STAGE 3: MULTIPROCESS S1 BATCH INFERENCE
    num_workers = min(os.cpu_count() or 4, 8)
    total_s1_records = 1732544
    batch_size = 1000

    logger.start_stage(3, 4, f"Processing {total_s1_records:,} S1 Batches (Multiprocessing: {num_workers} Workers)")
    t0_infer = time.time()
    peak_ram = get_ram_mb()

    # Launch worker pool
    pool = mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(target_attr, blocker))
    print(f"Worker pool active with {num_workers} processes.", flush=True)

    chunk_idx = 0
    subchunk_size = 100

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=batch_size, dtype=str):
        chunk_idx += 1
        
        if chunk_idx <= completed_batch_idx:
            continue

        eids = chunk['entity_id'].values
        raw_names = chunk['business_name'].values
        raw_addrs = chunk['business_address'].values
        countries = chunk['country'].fillna('UNKNOWN').astype(str).values

        batch_rows = list(zip(eids, raw_names, raw_addrs, countries))
        subchunks = [batch_rows[i:i + subchunk_size] for i in range(0, len(batch_rows), subchunk_size)]

        batch_cand_lines = []
        batch_pair_meta = [] # (s1_id, tid)
        batch_all_features = []
        batch_cand_count = 0

        # Run worker pool on subchunks
        pool_results = pool.map(_process_subchunk, subchunks)

        for res_list in pool_results:
            for s1_id, cand_str, valid_tids, feats in res_list:
                batch_cand_lines.append(f"{s1_id}\t{cand_str}\n")
                c_cnt = len(cand_str.split(",")) if cand_str else 0
                batch_cand_count += c_cnt
                
                for tid, feat in zip(valid_tids, feats):
                    batch_pair_meta.append((s1_id, tid))
                    batch_all_features.append(feat)

        total_s1_entities += len(batch_rows)
        total_candidate_pairs += batch_cand_count

        # Write candidate lines to disk
        with open(cand_file, "a", encoding="utf-8") as f_cand:
            f_cand.writelines(batch_cand_lines)
            f_cand.flush()

        # LightGBM scoring
        batch_matches = 0
        match_lines_dict = defaultdict(list)

        if batch_all_features:
            X_batch = np.array(batch_all_features, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            
            for (s1_id, tid), prob in zip(batch_pair_meta, probs):
                if prob >= 0.98:
                    match_lines_dict[s1_id].append(tid)

        batch_match_lines = []
        for s1_id in eids:
            m_list = match_lines_dict.get(s1_id, [])
            if m_list:
                batch_matches += len(m_list)
                match_str = ",".join(m_list)
            else:
                total_zero_match_s1 += 1
                match_str = ""
            batch_match_lines.append(f"{s1_id}\t{match_str}\n")

        total_predicted_matches += batch_matches

        # Write matching results to disk
        with open(match_file, "a", encoding="utf-8") as f_match:
            f_match.writelines(batch_match_lines)
            f_match.flush()

        # Save Checkpoint
        ckpt_data = {
            "completed_batches": chunk_idx,
            "processed_s1": total_s1_entities,
            "processed_candidates": total_candidate_pairs,
            "predicted_matches": total_predicted_matches,
            "zero_match_s1": total_zero_match_s1,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
            json.dump(ckpt_data, f_ckpt, indent=2)

        cur_ram = get_ram_mb()
        if cur_ram > peak_ram: peak_ram = cur_ram

        # Live Progress Logging
        logger.log_progress(
            current=total_s1_entities,
            total=total_s1_records,
            candidates=total_candidate_pairs,
            matches=total_predicted_matches,
            workers=num_workers,
            batch_num=chunk_idx
        )

        del batch_cand_lines, batch_pair_meta, batch_all_features, batch_match_lines
        gc.collect()

    pool.close()
    pool.join()

    t_infer_total = time.time() - t0_infer
    logger.finish_stage("S1 Batch Inference & Scoring")

    # STAGE 4: VERIFICATION & VALIDATION
    logger.start_stage(4, 4, "Verifying Outputs & Running Official Submission Validator")

    print(f"\nTotal Pipeline Runtime: {time.time() - t0_start:.2f} s", flush=True)
    print(f"Target Index Build Time: {t_index_build:.2f} s", flush=True)
    print(f"Inference & Scoring Time: {t_infer_total:.2f} s", flush=True)
    print(f"Total S1 Records Processed: {total_s1_entities:,}", flush=True)
    print(f"Total Candidate Pairs: {total_candidate_pairs:,}", flush=True)
    print(f"Total Matches (T>=0.98): {total_predicted_matches:,}", flush=True)
    print(f"Peak RAM: {peak_ram:.1f} MB", flush=True)

    cand_bytes = cand_file.stat().st_size if cand_file.exists() else 0
    match_bytes = match_file.stat().st_size if match_file.exists() else 0
    print(f"Output candidate_pairs.tsv Size: {cand_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"Output matching_results.tsv Size: {match_bytes / (1024*1024):.2f} MB", flush=True)

    # Run official validator
    import subprocess
    validator_script = base_dir / "utils" / "validate_submission.py"
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(test_dir)
    ]

    print("\nRunning Official Validator Command:", " ".join(cmd), flush=True)
    val_res = subprocess.run(cmd, capture_output=True, text=True)
    validator_output = val_res.stdout + "\n" + val_res.stderr
    print("=" * 80, flush=True)
    print("OFFICIAL VALIDATOR OUTPUT:\n", validator_output, flush=True)
    print("=" * 80, flush=True)

    if val_res.returncode == 0:
        print("\nSUBMISSION VALIDATION: PASSED SUCCESSFULLY!", flush=True)
    else:
        print("\nSUBMISSION VALIDATION: FAILED WITH ERRORS!", flush=True)

    logger.finish_stage("Verification & Validation")

if __name__ == '__main__':
    mp.freeze_support()
    main()
