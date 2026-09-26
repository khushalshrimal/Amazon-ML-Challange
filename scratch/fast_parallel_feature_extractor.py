"""
Multi-Threaded / Parallel Feature Extractor for 10k Validation Set.

Uses ProcessPoolExecutor with 8 parallel worker processes to extract 32 features
across 10,000 S1 records in <15 seconds, saving the resulting cache to scratch/val_features_cache_v2.joblib.
Continuous live telemetry is printed every batch.
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution" / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from test_blocking_experiments import AdvancedBlocker
from features_v2 import (
    OnDemandTargetStoreV2, precompute_s1_v2, fill_features_v2,
    FEATURE_NAMES_V2
)
from normalize import normalize_name, normalize_address, normalize_country

VAL_DIR = BASE_DIR / "scratch" / "val_split"
SCRATCH_DIR = BASE_DIR / "scratch"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

def process_s1_chunk(chunk_args):
    s1_chunk, target_raw_dict, blocker_index_data = chunk_args

    # Reconstruct blocker and target store locally in worker process
    target_store = OnDemandTargetStoreV2()
    target_store.raw_attr = target_raw_dict

    blocker = AdvancedBlocker(max_posting_size=100, use_prefix3=True)
    blocker.exact_name_idx, blocker.exact_addr_idx, blocker.name_pref_idx, blocker.addr_pref_idx, blocker.rare_tok_idx, blocker.stop_tokens = blocker_index_data

    chunk_results = []
    total_cands = 0

    for s1_rec in s1_chunk:
        s1_id = s1_rec["entity_id"]
        c = s1_rec.get("country", "")
        rn = s1_rec.get("business_name", "")
        ra = s1_rec.get("business_address", "")

        cands = blocker.get_candidates(s1_rec)
        total_cands += len(cands)

        s1_prep = precompute_s1_v2(s1_id, c, rn, ra)

        if cands:
            cands_list = list(cands)
            X = np.zeros((len(cands_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
            for r_idx, tid in enumerate(cands_list):
                fill_features_v2(X, r_idx, s1_prep, target_store, tid)
            chunk_results.append((s1_id, cands_list, X))
        else:
            chunk_results.append((s1_id, [], None))

    return chunk_results, total_cands

def main():
    print("=" * 70, flush=True)
    print("PARALLEL VALIDATION FEATURE EXTRACTION (8 CPU WORKERS)", flush=True)
    print("=" * 70, flush=True)
    start_time = time.time()

    # Step 1: Load Data
    val_s1_df = pd.read_csv(VAL_DIR / "val_s1.tsv", sep="\t", dtype=str).fillna("")
    val_targets_df = pd.read_csv(VAL_DIR / "val_targets.tsv", sep="\t", dtype=str).fillna("")

    s1_recs = val_s1_df.to_dict("records")
    target_recs = val_targets_df.to_dict("records")
    total_s1 = len(s1_recs)

    print(f"Validation S1 Count: {total_s1:,}", flush=True)
    print(f"Target Pool Count: {len(target_recs):,}", flush=True)

    # Step 2: Build Blocker Index & Target Store
    print("\n[STAGE 1/2] Building Blocker Index & Target Store...", flush=True)
    blocker = AdvancedBlocker(max_posting_size=100, use_prefix3=True)
    blocker.build_index(target_recs)

    target_raw_dict = {}
    for rec in target_recs:
        mid = rec["entity_id"]
        c = normalize_country(rec.get("country", ""))
        nn = normalize_name(rec.get("business_name", ""))
        na = normalize_address(rec.get("business_address", ""))
        target_raw_dict[mid] = (c, nn, na)

    blocker_index_data = (
        dict(blocker.exact_name_idx), dict(blocker.exact_addr_idx),
        dict(blocker.name_pref_idx), dict(blocker.addr_pref_idx),
        dict(blocker.rare_tok_idx), set(blocker.stop_tokens)
    )

    # Step 3: Split into chunks for 8 parallel workers
    num_workers = min(8, os.cpu_count() or 4)
    chunk_size = 500
    chunks = [s1_recs[i:i + chunk_size] for i in range(0, total_s1, chunk_size)]

    print(f"\n[STAGE 2/2] Launching {num_workers} parallel workers across {len(chunks)} chunks...", flush=True)

    s1_cands_list = []
    processed_count = 0
    total_candidates = 0
    last_print = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_s1_chunk, (chunk, target_raw_dict, blocker_index_data)): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            chunk_results, cands_count = future.result()
            s1_cands_list.extend(chunk_results)
            processed_count += len(chunk_results)
            total_candidates += cands_count

            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_s1 - processed_count) / rate if rate > 0 else 0
            pid, ram, cpu = get_telemetry()

            print(f"[PARALLEL FEATURE EXTRACTION] S1: {processed_count}/{total_s1} ({(processed_count/total_s1)*100:.1f}%) | "
                  f"Cands: {total_candidates:,} ({total_candidates/processed_count:.1f}/S1) | "
                  f"Rate: {rate:.1f} S1/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                  f"RAM: {ram:.2f}GB | CPU: {cpu}% | Workers: {num_workers} | PID: {pid}", flush=True)

    # Maintain deterministic S1 ordering
    s1_order_map = {rec['entity_id']: i for i, rec in enumerate(s1_recs)}
    s1_cands_list.sort(key=lambda x: s1_order_map[x[0]])

    cache_path = SCRATCH_DIR / "val_features_cache_v2.joblib"
    joblib.dump({'s1_cands_list': s1_cands_list}, cache_path)

    elapsed = time.time() - start_time
    pid, ram, cpu = get_telemetry()
    print("=" * 70, flush=True)
    print(f"PARALLEL FEATURE EXTRACTION COMPLETE!", flush=True)
    print(f"Saved: {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.1f} MB)", flush=True)
    print(f"Total Time: {elapsed:.2f}s | Throughput: {total_s1/elapsed:.1f} S1/s | RAM: {ram:.2f}GB", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
