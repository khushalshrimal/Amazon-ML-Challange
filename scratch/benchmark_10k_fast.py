import os
import sys
import time
import re
import joblib
import psutil
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from inference_optimized import fast_norm_name, fast_norm_addr, extract_digits, extract_ngrams

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')

# Load target records into compact dict
print("Loading target records into memory...", flush=True)
t0 = time.time()
target_raw = {}

s2_path = base_dir / "dataset" / "test" / "test_source2.tsv"
s3_path = base_dir / "dataset" / "test" / "test_source3.tsv"

for p in [s2_path, s3_path]:
    with open(p, "r", encoding="utf-8") as f:
        f.readline() # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, r_n, r_a, c = parts[0], parts[1], parts[2], parts[3]
                target_raw[eid] = (c, fast_norm_name(r_n), fast_norm_addr(r_a))

print(f"Loaded {len(target_raw):,} target records in {time.time()-t0:.2f}s.", flush=True)

# Parse store on-demand
target_cache = {}
def get_target_parsed(mid):
    if mid not in target_cache:
        c2, norm_n2, norm_a2 = target_raw[mid]
        toks_n2 = set(norm_n2.split())
        toks_a2 = set(norm_a2.split())
        ngrams_n2 = extract_ngrams(norm_n2, 3)
        ngrams_a2 = extract_ngrams(norm_a2, 3)
        digits_a2 = extract_digits(norm_a2)
        target_cache[mid] = (
            c2, norm_n2, norm_a2,
            toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2,
            len(norm_n2), len(norm_a2),
            len(toks_n2), len(toks_a2),
            len(ngrams_n2), len(ngrams_a2),
            len(digits_a2),
            1.0 if mid.startswith('S2-') else 0.0
        )
    return target_cache[mid]

# Benchmark scoring 10,000 S1 records
model_path = base_dir / "models" / "lightgbm_model.joblib"
model = joblib.load(model_path)

s1_path = base_dir / "dataset" / "test" / "test_source1.tsv"
cand_path = base_dir / "output" / "candidate_pairs.tsv"

s1_file = open(s1_path, "r", encoding="utf-8")
cand_file = open(cand_path, "r", encoding="utf-8")
s1_file.readline()
cand_file.readline()

print("\nBenchmarking single-thread feature extraction on 10,000 S1 records...", flush=True)
t_bench_start = time.time()

num_s1 = 10000
batch_X = np.empty((500000, 13), dtype=np.float32)
s1_processed = 0
cands_evaluated = 0
matches_found = 0

for _ in range(num_s1):
    s1_line = s1_file.readline()
    cand_line = cand_file.readline()
    if not s1_line or not cand_line:
        break

    s1_parts = s1_line.rstrip("\n").split("\t")
    cand_parts = cand_line.rstrip("\n").split("\t")
    if len(s1_parts) < 4:
        continue
    
    s1_id, r_n1, r_a1, c1 = s1_parts[0], s1_parts[1], s1_parts[2], s1_parts[3]
    cands = cand_parts[1].split(",") if len(cand_parts) > 1 and cand_parts[1] else []

    norm_n1 = fast_norm_name(r_n1)
    norm_a1 = fast_norm_addr(r_a1)
    toks_n1 = set(norm_n1.split())
    toks_a1 = set(norm_a1.split())
    ngrams_n1 = extract_ngrams(norm_n1, 3)
    ngrams_a1 = extract_ngrams(norm_a1, 3)
    digits_a1 = extract_digits(norm_a1)

    len_n1, len_a1 = len(norm_n1), len(norm_a1)
    len_tn1, len_ta1 = len(toks_n1), len(toks_a1)
    len_ng1_n, len_ng1_a = len(ngrams_n1), len(ngrams_a1)
    len_d1 = len(digits_a1)

    valid_tids = [tid for tid in cands if tid in target_raw]
    row_count = len(valid_tids)
    if row_count == 0:
        s1_processed += 1
        continue

    # Fill batch matrix
    for r_idx, tid in enumerate(valid_tids):
        t = get_target_parsed(tid)
        c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]
        len_n2, len_a2, len_tn2, len_ta2, len_ng2_n, len_ng2_a, len_d2, is_s2 = t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]

        n_exact = 1.0 if norm_n1 == norm_n2 and norm_n1 != "" else 0.0
        if len_tn1 > 0 and len_tn2 > 0:
            n_ov = len(toks_n1 & toks_n2)
            n_jac = n_ov / (len_tn1 + len_tn2 - n_ov)
        else: n_ov, n_jac = 0, 0.0

        if len_ng1_n > 0 and len_ng2_n > 0:
            ng_n_ov = len(ngrams_n1 & ngrams_n2)
            n_char_jac = ng_n_ov / (len_ng1_n + len_ng2_n - ng_n_ov)
        else: n_char_jac = 0.0

        if len_ta1 > 0 and len_ta2 > 0:
            a_ov = len(toks_a1 & toks_a2)
            a_jac = a_ov / (len_ta1 + len_ta2 - a_ov)
        else: a_ov, a_jac = 0, 0.0

        if len_ng1_a > 0 and len_ng2_a > 0:
            ng_a_ov = len(ngrams_a1 & ngrams_a2)
            a_char_jac = ng_a_ov / (len_ng1_a + len_ng2_a - ng_a_ov)
        else: a_char_jac = 0.0

        if len_d1 > 0 and len_d2 > 0:
            d_ov = len(digits_a1 & digits_a2)
            d_jac = d_ov / (len_d1 + len_d2 - d_ov)
        else: d_jac = 0.5

        c_match = 1.0 if c1 == c2 and c1 != "" else 0.0
        len_diff_n = float(abs(len_n1 - len_n2))
        len_diff_a = float(abs(len_a1 - len_a2))
        missing_addr = 1.0 if len_a1 == 0 or len_a2 == 0 else 0.0

        batch_X[r_idx, 0] = n_exact
        batch_X[r_idx, 1] = n_jac
        batch_X[r_idx, 2] = n_char_jac
        batch_X[r_idx, 3] = a_jac
        batch_X[r_idx, 4] = a_char_jac
        batch_X[r_idx, 5] = d_jac
        batch_X[r_idx, 6] = c_match
        batch_X[r_idx, 7] = len_diff_n
        batch_X[r_idx, 8] = len_diff_a
        batch_X[r_idx, 9] = float(n_ov)
        batch_X[r_idx, 10] = float(a_ov)
        batch_X[r_idx, 11] = missing_addr
        batch_X[r_idx, 12] = is_s2

    probs = model.predict_proba(batch_X[:row_count])[:, 1]
    matches = [tid for tid, p in zip(valid_tids, probs) if p >= 0.98]
    matches_found += len(matches)
    cands_evaluated += row_count
    s1_processed += 1

elapsed = time.time() - t_bench_start
rate = s1_processed / elapsed
cands_rate = cands_evaluated / elapsed
print(f"RESULTS FOR 10,000 S1:")
print(f"Elapsed: {elapsed:.2f}s | Speed: {rate:.1f} S1/sec | Candidates speed: {cands_rate:.1f} pairs/sec | Matches found: {matches_found}")
