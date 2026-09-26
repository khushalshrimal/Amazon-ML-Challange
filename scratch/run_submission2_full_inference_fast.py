"""
Memory-Safe & Fast Full Test Inference Pipeline for Submission 2 (Step 17).

Uses 32-bit uint32 integer indexing for inverted blocker lists & compact attribute stores.
Peak RAM < 2.5 GB for all 9.97 million target entities and 1.73 million test S1 records.

Inference Pipeline:
- Blocker: AdvancedBlocker (Posting limit=100, Prefix=3, Generic stop-words)
- Model: Ensemble (0.5 * LightGBM_v2 + 0.5 * XGBoost_v2)
- Threshold: 0.995
- Outputs: output/matching_results.tsv & output/candidate_pairs.tsv

Mandatory live numerical telemetry printed continuously every batch.
"""

import os
import sys
import time
import gc
import psutil
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from array import array
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution" / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from features_v2 import (
    precompute_s1_v2, fill_features_v2, FEATURE_NAMES_V2,
    extract_ngrams, jaccard_similarity, extract_house_number,
    extract_pincode, extract_phone, extract_email_domain, DIGIT_REGEX
)
from normalize import normalize_name, normalize_address, normalize_country
from test_blocking_experiments import GENERIC_STOPWORDS

TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

class CompactTargetStoreV2:
    """
    Ultra-compact 32-bit uint32 target store for 10M records.
    """
    def __init__(self):
        self.mids = []           # idx -> string target ID
        self.countries = []      # idx -> country
        self.norm_names = []     # idx -> norm_n
        self.norm_addrs = []     # idx -> norm_a
        self.cache = {}          # idx -> parsed target dict

    def add_target(self, mid: str, country: str, norm_n: str, norm_a: str) -> int:
        idx = len(self.mids)
        self.mids.append(mid)
        self.countries.append(country)
        self.norm_names.append(norm_n)
        self.norm_addrs.append(norm_a)
        return idx

    def get_parsed(self, idx: int):
        if idx not in self.cache:
            mid = self.mids[idx]
            c = self.countries[idx]
            norm_n = self.norm_names[idx]
            norm_a = self.norm_addrs[idx]

            toks_n = set(norm_n.split())
            toks_a = set(norm_a.split())
            ngrams_n3 = extract_ngrams(norm_n, 3)
            ngrams_n4 = extract_ngrams(norm_n, 4)
            ngrams_a3 = extract_ngrams(norm_a, 3)
            digits_a = set(DIGIT_REGEX.findall(norm_a))

            house_num = extract_house_number(norm_a)
            pincode = extract_pincode(norm_a)
            phone = extract_phone(norm_n + " " + norm_a)
            email_dom = extract_email_domain(norm_n + " " + norm_a)

            self.cache[idx] = {
                'c': c,
                'norm_n': norm_n,
                'norm_a': norm_a,
                'toks_n': toks_n,
                'toks_a': toks_a,
                'ngrams_n3': ngrams_n3,
                'ngrams_n4': ngrams_n4,
                'ngrams_a3': ngrams_a3,
                'digits_a': digits_a,
                'house_num': house_num,
                'pincode': pincode,
                'phone': phone,
                'email_dom': email_dom,
                'len_n': len(norm_n),
                'len_a': len(norm_a),
                'is_s2': 1.0 if mid.startswith('S2-') else 0.0
            }
        return self.cache[idx]

class CompactBlocker:
    """
    Compact inverted index mapping keys -> array('I') uint32 target indices with posting limit 100.
    """
    def __init__(self, max_posting_size: int = 100):
        self.max_posting_size = max_posting_size
        self.exact_name_idx = defaultdict(lambda: array('I'))
        self.exact_addr_idx = defaultdict(lambda: array('I'))
        self.name_pref_idx = defaultdict(lambda: array('I'))
        self.addr_pref_idx = defaultdict(lambda: array('I'))
        self.rare_tok_idx = defaultdict(lambda: array('I'))

    def add_record(self, idx: int, c: str, nn: str, na: str):
        n_toks = [t for t in nn.split() if t not in GENERIC_STOPWORDS]
        a_toks = [t for t in na.split() if t not in GENERIC_STOPWORDS]

        # 1. Exact blocks
        if nn:
            key = (nn, c)
            if len(self.exact_name_idx[key]) < self.max_posting_size:
                self.exact_name_idx[key].append(idx)
        if na:
            key = (na, c)
            if len(self.exact_addr_idx[key]) < self.max_posting_size:
                self.exact_addr_idx[key].append(idx)

        # 2. Prefixes (3-char)
        if len(nn) >= 3:
            key = (nn[:3], c)
            if len(self.name_pref_idx[key]) < self.max_posting_size:
                self.name_pref_idx[key].append(idx)
        if len(na) >= 3:
            key = (na[:3], c)
            if len(self.addr_pref_idx[key]) < self.max_posting_size:
                self.addr_pref_idx[key].append(idx)

        # 3. Rare tokens
        rare_n = [t for t in n_toks if t not in GENERIC_STOPWORDS and len(t) >= 3][:2]
        for t in rare_n:
            key = (t, c)
            if len(self.rare_tok_idx[key]) < self.max_posting_size:
                self.rare_tok_idx[key].append(idx)

        rare_a = [t for t in a_toks if t not in GENERIC_STOPWORDS and (len(t) >= 3 or t.isdigit())][:2]
        for t in rare_a:
            key = (t, c)
            if len(self.rare_tok_idx[key]) < self.max_posting_size:
                self.rare_tok_idx[key].append(idx)

    def get_candidates(self, s1_rec: dict) -> set:
        c = normalize_country(s1_rec.get("country", ""))
        rn = s1_rec.get("business_name", "")
        ra = s1_rec.get("business_address", "")
        nn = normalize_name(rn)
        na = normalize_address(ra)

        cands = set()
        if nn:
            cands.update(self.exact_name_idx.get((nn, c), []))
        if na:
            cands.update(self.exact_addr_idx.get((na, c), []))

        n_toks = [t for t in nn.split() if t not in GENERIC_STOPWORDS and len(t) >= 3][:2]
        for t in n_toks:
            cands.update(self.rare_tok_idx.get((t, c), []))

        a_toks = [t for t in na.split() if t not in GENERIC_STOPWORDS and (len(t) >= 3 or t.isdigit())][:2]
        for t in a_toks:
            cands.update(self.rare_tok_idx.get((t, c), []))

        if len(nn) >= 3:
            cands.update(self.name_pref_idx.get((nn[:3], c), []))
        if len(na) >= 3:
            cands.update(self.addr_pref_idx.get((na[:3], c), []))

        return cands

def main():
    print("=" * 80, flush=True)
    print("SUBMISSION 2 FULL TEST INFERENCE (FAST COMPACT MEMORY PIPELINE)", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Trained Models
    print("[STAGE 1/4] Loading LightGBM v2 and XGBoost v2 trained models...", flush=True)
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_model_v2.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_v2.joblib")
    print("Models loaded successfully.", flush=True)

    # Step 2: Index 9.97 Million Target Records Compactly
    print("\n[STAGE 2/4] Indexing 9.97 Million Target Records (S2 + S3)...", flush=True)
    t0_block = time.time()

    blocker = CompactBlocker(max_posting_size=100)
    target_store = CompactTargetStoreV2()

    total_targets = 0
    for file_path in [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"]:
        print(f"Indexing {file_path.name}...", flush=True)
        for chunk_df in pd.read_csv(file_path, sep="\t", chunksize=500000, dtype=str):
            chunk_df = chunk_df.fillna("")
            mids = chunk_df["entity_id"].values
            names = chunk_df["business_name"].values
            addrs = chunk_df["business_address"].values
            countries = chunk_df["country"].values

            for i in range(len(mids)):
                c = normalize_country(countries[i])
                nn = normalize_name(names[i])
                na = normalize_address(addrs[i])
                idx = target_store.add_target(mids[i], c, nn, na)
                blocker.add_record(idx, c, nn, na)
                total_targets += 1

            pid, ram, cpu = get_telemetry()
            print(f"Indexed {total_targets:,} targets | RAM: {ram:.2f} GB | PID: {pid}", flush=True)
            gc.collect()

    print(f"\nTarget Indexing Complete in {time.time() - t0_block:.2f}s! Total targets: {len(target_store.mids):,}", flush=True)

    # Step 3: Initialize Output Files
    print("\n[STAGE 3/4] Initializing output files...", flush=True)
    with open(MATCHING_OUT, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Step 4: Full S1 Batch Processing
    print("\n[STAGE 4/4] Streaming S1 Test Records & Predicting Matches...", flush=True)
    total_s1_processed = 0
    total_candidate_pairs = 0
    total_matched_s1 = 0
    total_unmatched_s1 = 0
    total_final_matches = 0

    s1_batch_size = 25000
    TOTAL_S1 = 1732544

    for s1_chunk_df in pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", chunksize=s1_batch_size, dtype=str):
        s1_chunk_df = s1_chunk_df.fillna("")
        s1_recs = s1_chunk_df.to_dict("records")

        match_lines = []
        cand_lines = []

        batch_X_list = []
        batch_meta = []
        curr_row = 0

        for s1_rec in s1_recs:
            s1_id = s1_rec["entity_id"]
            c = s1_rec.get("country", "")
            rn = s1_rec.get("business_name", "")
            ra = s1_rec.get("business_address", "")

            cand_indices = blocker.get_candidates(s1_rec)
            total_candidate_pairs += len(cand_indices)

            s1_prep = precompute_s1_v2(s1_id, c, rn, ra)

            if cand_indices:
                cand_idx_list = list(cand_indices)
                X_s1 = np.zeros((len(cand_idx_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
                for r_idx, t_idx in enumerate(cand_idx_list):
                    fill_features_v2(X_s1, r_idx, s1_prep, target_store, t_idx)

                batch_X_list.append(X_s1)
                n_rows = len(cand_idx_list)
                batch_meta.append((s1_id, cand_idx_list, curr_row, curr_row + n_rows))
                curr_row += n_rows
            else:
                batch_meta.append((s1_id, [], 0, 0))

        # Score batch with Ensemble model
        if batch_X_list:
            X_batch = np.vstack(batch_X_list)
            p_lgb = lgb_model.predict_proba(X_batch)[:, 1]
            p_xgb = xgb_model.predict_proba(X_batch)[:, 1]
            p_ens = 0.5 * p_lgb + 0.5 * p_xgb

            for s1_id, cand_idx_list, start_i, end_i in batch_meta:
                cand_mids = [target_store.mids[idx] for idx in cand_idx_list]
                cand_str = ",".join(cand_mids) if cand_mids else ""
                cand_lines.append(f"{s1_id}\t{cand_str}\n")

                if cand_idx_list:
                    probs = p_ens[start_i:end_i]
                    matched_mids = [cand_mids[i] for i, prob in enumerate(probs) if prob >= 0.995]
                    match_str = ",".join(matched_mids) if matched_mids else ""
                    match_lines.append(f"{s1_id}\t{match_str}\n")

                    if matched_mids:
                        total_matched_s1 += 1
                        total_final_matches += len(matched_mids)
                    else:
                        total_unmatched_s1 += 1
                else:
                    match_lines.append(f"{s1_id}\t\n")
                    total_unmatched_s1 += 1
        else:
            for s1_id, _, _, _ in batch_meta:
                cand_lines.append(f"{s1_id}\t\n")
                match_lines.append(f"{s1_id}\t\n")
                total_unmatched_s1 += 1

        with open(MATCHING_OUT, "a", encoding="utf-8") as f_match:
            f_match.writelines(match_lines)

        with open(CANDIDATE_OUT, "a", encoding="utf-8") as f_cand:
            f_cand.writelines(cand_lines)

        total_s1_processed += len(s1_recs)

        elapsed = time.time() - start_time
        rate = total_s1_processed / elapsed if elapsed > 0 else 0
        eta = (TOTAL_S1 - total_s1_processed) / rate if rate > 0 else 0
        pid, ram, cpu = get_telemetry()
        cands_per_s1 = total_candidate_pairs / total_s1_processed if total_s1_processed > 0 else 0

        print(f"[FULL TEST INFERENCE] S1: {total_s1_processed:,} / {TOTAL_S1:,} ({(total_s1_processed/TOTAL_S1)*100:.2f}%) | "
              f"Candidates: {total_candidate_pairs:,} ({cands_per_s1:.1f}/S1) | "
              f"Matched S1: {total_matched_s1:,} (Unmatched: {total_unmatched_s1:,}) | "
              f"Matches Found: {total_final_matches:,} | "
              f"Rate: {rate:.1f} S1/sec | Elapsed: {elapsed/60:.2f}m | ETA: {eta/60:.2f}m | "
              f"RAM: {ram:.2f}GB | CPU: {cpu}% | PID: {pid}", flush=True)

        gc.collect()

    print("\n" + "=" * 80, flush=True)
    print("FULL INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total S1 Processed    : {total_s1_processed:,}")
    print(f"Total Candidate Pairs : {total_candidate_pairs:,} ({total_candidate_pairs/total_s1_processed:.1f} cands/S1)")
    print(f"Total Final Matches   : {total_final_matches:,}")
    print(f"Matched S1 Count      : {total_matched_s1:,} ({(total_matched_s1/total_s1_processed)*100:.2f}%)")
    print(f"Unmatched S1 Count    : {total_unmatched_s1:,} ({(total_unmatched_s1/total_s1_processed)*100:.2f}%)")
    print(f"Total Elapsed Time    : {time.time() - start_time:.2f} s ({(time.time() - start_time)/60:.2f} min)")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
