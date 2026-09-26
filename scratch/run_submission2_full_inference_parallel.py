import sys
import os
import time
import json
import gc
import psutil
import joblib
from pathlib import Path
from array import array
from collections import defaultdict
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution" / "src"))

from normalize import normalize_name, normalize_address, normalize_country
from features_v2 import (
    precompute_s1_v2, FEATURE_NAMES_V2,
    extract_ngrams, extract_house_number, extract_pincode,
    extract_phone, extract_email_domain, DIGIT_REGEX
)

TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"
CHECKPOINT_FILE = SCRATCH_DIR / "inference_checkpoint.json"

GENERIC_STOPWORDS = {
    'private', 'limited', 'pvt', 'ltd', 'inc', 'corp', 'corporation',
    'llc', 'co', 'company', 'enterprises', 'group', 'services',
    'india', 'door', 'plot', 'street', 'road', 'near', 'opp', 'opposite',
    'behind', 'flat', 'floor', 'building', 'no', 'number', 'main', 'cross'
}

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

class CompactTargetStoreV2:
    def __init__(self):
        self.mids = []
        self.countries = []
        self.norm_names = []
        self.norm_addrs = []

    def add_target(self, mid: str, country: str, norm_n: str, norm_a: str) -> int:
        idx = len(self.mids)
        self.mids.append(mid)
        self.countries.append(country)
        self.norm_names.append(norm_n)
        self.norm_addrs.append(norm_a)
        return idx

    def get_raw_tuple(self, idx: int):
        return (self.mids[idx], self.countries[idx], self.norm_names[idx], self.norm_addrs[idx])

class CompactBlocker:
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

        if nn:
            key = (nn, c)
            if len(self.exact_name_idx[key]) < self.max_posting_size:
                self.exact_name_idx[key].append(idx)
        if na:
            key = (na, c)
            if len(self.exact_addr_idx[key]) < self.max_posting_size:
                self.exact_addr_idx[key].append(idx)

        rare_n = [t for t in n_toks if len(t) >= 3][:2]
        for t in rare_n:
            key = (t, c)
            if len(self.rare_tok_idx[key]) < self.max_posting_size:
                self.rare_tok_idx[key].append(idx)

        rare_a = [t for t in a_toks if len(t) >= 3 or t.isdigit()][:2]
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

# Worker state
_GLOBAL_LGB_MODEL = None
_GLOBAL_XGB_MODEL = None
_WORKER_CACHE = {}

def _init_worker(models_dir_str):
    global _GLOBAL_LGB_MODEL, _GLOBAL_XGB_MODEL, _WORKER_CACHE
    p_models = Path(models_dir_str)
    _GLOBAL_LGB_MODEL = joblib.load(p_models / "lightgbm_model_v2.joblib")
    _GLOBAL_XGB_MODEL = joblib.load(p_models / "xgboost_model_v2.joblib")
    _WORKER_CACHE = {}

def _parse_target_raw(raw_tuple):
    mid, c, norm_n, norm_a = raw_tuple
    if mid not in _WORKER_CACHE:
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

        _WORKER_CACHE[mid] = {
            'mid': mid,
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
    return _WORKER_CACHE[mid]

def _worker_process_chunk(chunk_items):
    """
    Worker processes a list of items: (s1_prep, target_raw_tuples)
    """
    results = []
    for s1_prep, target_raw_tuples in chunk_items:
        s1_id = s1_prep['s1_id']
        cand_mids = [t[0] for t in target_raw_tuples]

        if not target_raw_tuples:
            results.append((s1_id, [], []))
            continue

        target_parsed_list = [_parse_target_raw(t_raw) for t_raw in target_raw_tuples]

        X = np.zeros((len(target_parsed_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
        for r_idx, t_dict in enumerate(target_parsed_list):
            n1, n2 = s1_prep['norm_n'], t_dict['norm_n']
            name_exact = 1.0 if n1 and n1 == n2 else 0.0
            name_jaccard = float(len(s1_prep['toks_n'] & t_dict['toks_n'])) / len(s1_prep['toks_n'] | t_dict['toks_n']) if (s1_prep['toks_n'] and t_dict['toks_n']) else 0.0
            name_overlap = float(len(s1_prep['toks_n'] & t_dict['toks_n']))

            # Fast-path RapidFuzz early pruning
            if name_jaccard == 0.0 and not name_exact and s1_prep['ngrams_n3'].isdisjoint(t_dict['ngrams_n3']):
                rf_ratio, part_ratio, w_ratio, tok_set_ratio = 0.0, 0.0, 0.0, 0.0
            else:
                from rapidfuzz import fuzz
                rf_ratio = fuzz.ratio(n1, n2) / 100.0
                part_ratio = fuzz.partial_ratio(n1, n2) / 100.0
                w_ratio = fuzz.WRatio(n1, n2) / 100.0
                tok_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0

            n3_jaccard = float(len(s1_prep['ngrams_n3'] & t_dict['ngrams_n3'])) / len(s1_prep['ngrams_n3'] | t_dict['ngrams_n3']) if (s1_prep['ngrams_n3'] and t_dict['ngrams_n3']) else 0.0
            n4_jaccard = float(len(s1_prep['ngrams_n4'] & t_dict['ngrams_n4'])) / len(s1_prep['ngrams_n4'] | t_dict['ngrams_n4']) if (s1_prep['ngrams_n4'] and t_dict['ngrams_n4']) else 0.0

            pref3_match = 1.0 if n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3 else 0.0
            pref4_match = 1.0 if n1[:4] == n2[:4] and len(n1) >= 4 and len(n2) >= 4 else 0.0

            len_diff_n = float(abs(s1_prep['len_n'] - t_dict['len_n']))
            max_len_n = max(s1_prep['len_n'], t_dict['len_n'])
            len_ratio_n = (min(s1_prep['len_n'], t_dict['len_n']) / max_len_n) if max_len_n > 0 else 1.0

            a1, a2 = s1_prep['norm_a'], t_dict['norm_a']
            addr_exact = 1.0 if a1 and a1 == a2 else 0.0
            addr_jaccard = float(len(s1_prep['toks_a'] & t_dict['toks_a'])) / len(s1_prep['toks_a'] | t_dict['toks_a']) if (s1_prep['toks_a'] and t_dict['toks_a']) else 0.0
            addr_overlap = float(len(s1_prep['toks_a'] & t_dict['toks_a']))

            if addr_jaccard == 0.0 and not addr_exact:
                addr_rf_ratio, addr_tok_set_ratio = 0.0, 0.0
            else:
                from rapidfuzz import fuzz
                addr_rf_ratio = fuzz.ratio(a1, a2) / 100.0
                addr_tok_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0

            a3_jaccard = float(len(s1_prep['ngrams_a3'] & t_dict['ngrams_a3'])) / len(s1_prep['ngrams_a3'] | t_dict['ngrams_a3']) if (s1_prep['ngrams_a3'] and t_dict['ngrams_a3']) else 0.0
            digit_jaccard = float(len(s1_prep['digits_a'] & t_dict['digits_a'])) / len(s1_prep['digits_a'] | t_dict['digits_a']) if (s1_prep['digits_a'] and t_dict['digits_a']) else 0.0

            house_num_match = 1.0 if s1_prep['house_num'] == t_dict['house_num'] else 0.0 if (s1_prep['house_num'] and t_dict['house_num']) else 0.5
            pincode_match = 1.0 if s1_prep['pincode'] == t_dict['pincode'] else 0.0 if (s1_prep['pincode'] and t_dict['pincode']) else 0.5
            len_diff_a = float(abs(s1_prep['len_a'] - t_dict['len_a']))

            phone_match = 1.0 if s1_prep['phone'] == t_dict['phone'] else 0.0 if (s1_prep['phone'] and t_dict['phone']) else 0.5
            email_dom_match = 1.0 if s1_prep['email_dom'] == t_dict['email_dom'] else 0.0 if (s1_prep['email_dom'] and t_dict['email_dom']) else 0.5

            c1, c2 = s1_prep['c'], t_dict['c']
            country_match = 1.0 if c1 and c1 == c2 else 0.0
            is_s2 = t_dict['is_s2']

            name_x_c = rf_ratio * country_match
            addr_x_c = addr_rf_ratio * country_match
            name_x_addr = rf_ratio * addr_rf_ratio
            jaccard_x = name_jaccard * addr_jaccard

            X[r_idx, :] = [
                name_exact, name_jaccard, name_overlap,
                rf_ratio, part_ratio, w_ratio, tok_set_ratio,
                n3_jaccard, n4_jaccard, pref3_match, pref4_match,
                len_diff_n, len_ratio_n,
                addr_exact, addr_jaccard, addr_overlap,
                addr_rf_ratio, addr_tok_set_ratio,
                a3_jaccard, digit_jaccard, house_num_match, pincode_match, len_diff_a,
                phone_match, email_dom_match,
                country_match, is_s2,
                name_x_c, addr_x_c, name_x_addr, jaccard_x, 1.0
            ]

        p_lgb = _GLOBAL_LGB_MODEL.predict_proba(X)[:, 1]
        p_xgb = _GLOBAL_XGB_MODEL.predict_proba(X)[:, 1]
        p_ens = 0.5 * p_lgb + 0.5 * p_xgb

        matched_mids = [cand_mids[i] for i, prob in enumerate(p_ens) if prob >= 0.995]
        results.append((s1_id, cand_mids, matched_mids))

    return results

def main():
    print("=" * 80, flush=True)
    print("SUBMISSION 2 ULTRA-FAST PARALLEL INFERENCE PIPELINE (RESUMABLE)", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Target Store & Inverted Index Building
    print("\n[STAGE 1/2] Indexing 9.97 Million Target Records (S2 + S3)...", flush=True)
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

    print(f"Target Indexing Complete in {time.time() - t0_block:.2f}s! Total targets: {len(target_store.mids):,}", flush=True)

    # Checkpoint setup
    start_s1_idx = 0
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f_ck:
                ck_data = json.load(f_ck)
                start_s1_idx = ck_data.get("processed_s1", 0)
                print(f"Resuming from Checkpoint: {start_s1_idx:,} S1 records completed.", flush=True)
        except Exception:
            start_s1_idx = 0

    if start_s1_idx == 0:
        with open(MATCHING_OUT, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
        with open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Step 2: Parallel Batch Inference
    NUM_WORKERS = 6
    BATCH_SIZE = 5000
    WORKER_SUBCHUNK = 500
    TOTAL_S1 = 1732544

    print(f"\n[STAGE 2/2] Streaming S1 Test Records & Parallel Inference with {NUM_WORKERS} CPU Workers...", flush=True)

    total_s1_processed = start_s1_idx
    total_candidate_pairs = 0
    total_matched_s1 = 0
    total_unmatched_s1 = 0
    total_final_matches = 0

    s1_chunk_reader = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", chunksize=BATCH_SIZE, dtype=str)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker, initargs=(str(MODELS_DIR),)) as executor:
        curr_s1_count = 0
        for s1_chunk_df in s1_chunk_reader:
            chunk_len = len(s1_chunk_df)
            if curr_s1_count + chunk_len <= start_s1_idx:
                curr_s1_count += chunk_len
                continue

            s1_chunk_df = s1_chunk_df.fillna("")
            s1_recs = s1_chunk_df.to_dict("records")

            worker_items = []
            for s1_rec in s1_recs:
                s1_id = s1_rec["entity_id"]
                c = s1_rec.get("country", "")
                rn = s1_rec.get("business_name", "")
                ra = s1_rec.get("business_address", "")

                cand_indices = blocker.get_candidates(s1_rec)
                total_candidate_pairs += len(cand_indices)

                s1_prep = precompute_s1_v2(s1_id, c, rn, ra)
                raw_target_tuples = [target_store.get_raw_tuple(idx) for idx in cand_indices]
                worker_items.append((s1_prep, raw_target_tuples))

            subchunks = [worker_items[i:i + WORKER_SUBCHUNK] for i in range(0, len(worker_items), WORKER_SUBCHUNK)]
            futures = [executor.submit(_worker_process_chunk, sc) for sc in subchunks]

            ordered_sub_results = [f.result() for f in futures]

            match_lines = []
            cand_lines = []

            for sub_res in ordered_sub_results:
                for s1_id, cand_mids, matched_mids in sub_res:
                    cand_str = ",".join(cand_mids) if cand_mids else ""
                    cand_lines.append(f"{s1_id}\t{cand_str}\n")

                    match_str = ",".join(matched_mids) if matched_mids else ""
                    match_lines.append(f"{s1_id}\t{match_str}\n")

                    if matched_mids:
                        total_matched_s1 += 1
                        total_final_matches += len(matched_mids)
                    else:
                        total_unmatched_s1 += 1

            # Append outputs
            with open(MATCHING_OUT, "a", encoding="utf-8") as f_match:
                f_match.writelines(match_lines)

            with open(CANDIDATE_OUT, "a", encoding="utf-8") as f_cand:
                f_cand.writelines(cand_lines)

            total_s1_processed += chunk_len
            curr_s1_count += chunk_len

            # Update checkpoint
            with open(CHECKPOINT_FILE, "w") as f_ck:
                json.dump({"processed_s1": total_s1_processed}, f_ck)

            # Telemetry
            elapsed = time.time() - start_time
            rate = total_s1_processed / elapsed if elapsed > 0 else 0
            eta = (TOTAL_S1 - total_s1_processed) / rate if rate > 0 else 0
            pid, ram, cpu = get_telemetry()
            cands_per_s1 = total_candidate_pairs / total_s1_processed if total_s1_processed > 0 else 0
            match_file_size_mb = MATCHING_OUT.stat().st_size / (1024 * 1024) if MATCHING_OUT.exists() else 0

            print(f"[FULL TEST INFERENCE] S1: {total_s1_processed:,} / {TOTAL_S1:,} ({(total_s1_processed/TOTAL_S1)*100:.2f}%) | "
                  f"Candidates: {total_candidate_pairs:,} ({cands_per_s1:.1f}/S1) | "
                  f"Matched S1: {total_matched_s1:,} (Unmatched: {total_unmatched_s1:,}) | "
                  f"Matches Found: {total_final_matches:,} | "
                  f"Rate: {rate:.1f} S1/sec | Elapsed: {elapsed/60:.2f}m | ETA: {eta/60:.2f}m | "
                  f"RAM: {ram:.2f}GB | CPU: {cpu}% | Workers: {NUM_WORKERS} | PID: {pid} | Output: {match_file_size_mb:.1f}MB", flush=True)

            gc.collect()

    print("\n" + "=" * 80, flush=True)
    print("PARALLEL INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total S1 Processed    : {total_s1_processed:,}")
    print(f"Total Candidate Pairs : {total_candidate_pairs:,}")
    print(f"Total Final Matches   : {total_final_matches:,}")
    print(f"Matched S1 Count      : {total_matched_s1:,} ({(total_matched_s1/total_s1_processed)*100:.2f}%)")
    print(f"Total Elapsed Time    : {(time.time() - start_time)/60:.2f} minutes")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
