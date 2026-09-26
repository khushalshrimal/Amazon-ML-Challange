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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution" / "src"))

from normalize import normalize_name, normalize_address, normalize_country
from features_v2 import (
    precompute_s1_v2, fill_features_v2, FEATURE_NAMES_V2,
    extract_ngrams, extract_house_number, extract_pincode,
    extract_phone, extract_email_domain, DIGIT_REGEX
)

VAL_DIR = PROJECT_ROOT / "scratch" / "val_split"
VAL_S1 = VAL_DIR / "val_s1.tsv"
VAL_TARGETS = VAL_DIR / "val_targets.tsv"
VAL_GT = VAL_DIR / "val_gt.tsv"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

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

class CheapTargetStore:
    def __init__(self):
        self.mids = []
        self.countries = []
        self.norm_names = []
        self.norm_addrs = []
        self.cache = {}

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
        return self.cache[idx]

class CheapBlocker:
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

    def get_cheap_ranked_candidates(self, s1_prep: dict, target_store: CheapTargetStore, top_k: int = 50) -> list:
        c = s1_prep['c']
        nn = s1_prep['norm_n']
        na = s1_prep['norm_a']

        cand_scores = defaultdict(float)

        if nn:
            for idx in self.exact_name_idx.get((nn, c), []):
                cand_scores[idx] += 100.0
        if na:
            for idx in self.exact_addr_idx.get((na, c), []):
                cand_scores[idx] += 50.0

        n_toks = [t for t in nn.split() if t not in GENERIC_STOPWORDS and len(t) >= 3][:2]
        for t in n_toks:
            for idx in self.rare_tok_idx.get((t, c), []):
                cand_scores[idx] += 8.0

        a_toks = [t for t in na.split() if t not in GENERIC_STOPWORDS and (len(t) >= 3 or t.isdigit())][:2]
        for t in a_toks:
            for idx in self.rare_tok_idx.get((t, c), []):
                cand_scores[idx] += 4.0

        if len(nn) >= 3:
            for idx in self.name_pref_idx.get((nn[:3], c), []):
                cand_scores[idx] += 10.0
        if len(na) >= 3:
            for idx in self.addr_pref_idx.get((na[:3], c), []):
                cand_scores[idx] += 5.0

        if not cand_scores:
            return []

        if top_k and len(cand_scores) > top_k:
            sorted_cand_indices = sorted(cand_scores.keys(), key=lambda k: cand_scores[k], reverse=True)[:top_k]
        else:
            sorted_cand_indices = sorted(cand_scores.keys(), key=lambda k: cand_scores[k], reverse=True)

        return sorted_cand_indices

def compute_macro_f05(pred_dict, gt_dict):
    precisions = []
    recalls = []
    f05s = []

    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())

        if not gt_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f05s.append(1.0)
            continue
        elif not gt_set and pred_set:
            precisions.append(0.0)
            recalls.append(1.0)
            f05s.append(0.0)
            continue
        elif gt_set and not pred_set:
            precisions.append(1.0)
            recalls.append(0.0)
            f05s.append(0.0)
            continue

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec > 0:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        else:
            f05 = 0.0

        precisions.append(prec)
        recalls.append(rec)
        f05s.append(f05)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))
    return macro_f05, macro_prec, macro_rec

def run_benchmark():
    print("=" * 80)
    print("FAST TOP-K TWO-STAGE RETRIEVAL BENCHMARK (K = 100, 50, 30, 20, 10)")
    print("=" * 80)

    print("\nLoading Models...")
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_model_v2.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_v2.joblib")

    print("\nLoading Ground Truth TSV...")
    val_gt_df = pd.read_csv(VAL_GT, sep="\t", dtype=str).fillna("")
    val_gt = {}
    for _, row in val_gt_df.iterrows():
        s1 = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        val_gt[s1] = set(m_str.split(",")) if m_str else set()

    print(f"Loaded GT for {len(val_gt):,} S1 entities.")

    print("\nLoading Validation Targets (434,567 entities)...")
    t0_targets = time.time()
    target_store = CheapTargetStore()
    blocker = CheapBlocker(max_posting_size=100)

    val_targets_df = pd.read_csv(VAL_TARGETS, sep="\t", dtype=str).fillna("")
    mids = val_targets_df["entity_id"].values
    names = val_targets_df["business_name"].values
    addrs = val_targets_df["business_address"].values
    countries = val_targets_df["country"].values

    for i in range(len(mids)):
        c = normalize_country(countries[i])
        nn = normalize_name(names[i])
        na = normalize_address(addrs[i])
        idx = target_store.add_target(mids[i], c, nn, na)
        blocker.add_record(idx, c, nn, na)

    print(f"Target indexing complete in {time.time() - t0_targets:.2f}s!")

    print("\nLoading Validation S1 Records (10,000 entities)...")
    val_s1_df = pd.read_csv(VAL_S1, sep="\t", dtype=str).fillna("")
    val_s1_recs = val_s1_df.to_dict("records")

    s1_preps = []
    for r in val_s1_recs:
        prep = precompute_s1_v2(r["entity_id"], r.get("country", ""), r.get("business_name", ""), r.get("business_address", ""))
        s1_preps.append(prep)

    k_values = [100, 50, 30, 20, 10]
    benchmark_results = {}

    for K in k_values:
        label = f"Top K={K}"
        print(f"\n" + "-" * 60)
        print(f"EVALUATING STAGE A + B BENCHMARK: {label}")
        print("-" * 60)

        t0 = time.time()
        total_gt_pairs = 0
        retained_gt_pairs = 0

        cand_counts = []
        pred_dict = {}

        total_scoring_pairs = 0

        for prep in s1_preps:
            s1_id = prep['s1_id']
            gt_set = val_gt.get(s1_id, set())
            total_gt_pairs += len(gt_set)

            cand_indices = blocker.get_cheap_ranked_candidates(prep, target_store, top_k=K)
            cand_counts.append(len(cand_indices))

            cand_mids = set(target_store.mids[idx] for idx in cand_indices)
            retained_gt_pairs += len(gt_set & cand_mids)

            if not cand_indices:
                pred_dict[s1_id] = set()
                continue

            total_scoring_pairs += len(cand_indices)
            target_parsed_list = [target_store.get_parsed(idx) for idx in cand_indices]

            X = np.zeros((len(target_parsed_list), len(FEATURE_NAMES_V2)), dtype=np.float32)
            for r_idx, target_idx in enumerate(cand_indices):
                fill_features_v2(X, r_idx, prep, target_store, target_idx)

            p_lgb = lgb_model.predict_proba(X)[:, 1]
            p_xgb = xgb_model.predict_proba(X)[:, 1]
            p_ens = 0.5 * p_lgb + 0.5 * p_xgb

            matched_mids = set(t_dict['mid'] for i, t_dict in enumerate(target_parsed_list) if p_ens[i] >= 0.995)
            pred_dict[s1_id] = matched_mids

        elapsed = time.time() - t0
        cand_recall = (retained_gt_pairs / total_gt_pairs) * 100 if total_gt_pairs > 0 else 0
        avg_cands = np.mean(cand_counts)
        max_cands = np.max(cand_counts)

        macro_f05, macro_prec, macro_rec = compute_macro_f05(pred_dict, val_gt)

        ms_per_s1 = (elapsed / len(s1_preps)) * 1000
        est_full_test_hours = (elapsed / len(s1_preps)) * 1732544 / 3600

        pid, ram, cpu = get_telemetry()

        res = {
            "label": label,
            "K": K,
            "candidate_recall": f"{cand_recall:.2f}%",
            "avg_candidates_per_s1": f"{avg_cands:.1f}",
            "max_candidates_per_s1": int(max_cands),
            "macro_f05": f"{macro_f05:.4f}",
            "macro_precision": f"{macro_prec:.4f}",
            "macro_recall": f"{macro_rec:.4f}",
            "runtime_10k_s1_sec": f"{elapsed:.2f}s",
            "ms_per_s1": f"{ms_per_s1:.2f}ms",
            "est_full_test_hours": f"{est_full_test_hours:.2f}h",
            "total_scored_pairs": total_scoring_pairs,
            "ram_gb": f"{ram:.2f}GB",
            "cpu_percent": f"{cpu}%"
        }
        benchmark_results[label] = res

        print(f"Candidate Recall      : {cand_recall:.2f}%")
        print(f"Avg Candidates / S1   : {avg_cands:.1f} (Max: {max_cands})")
        print(f"Macro F0.5            : {macro_f05:.4f} (Precision: {macro_prec:.4f}, Recall: {macro_rec:.4f})")
        print(f"Runtime (10k S1)      : {elapsed:.2f}s ({ms_per_s1:.2f} ms/S1)")
        print(f"Est Full Test Runtime : {est_full_test_hours:.2f} hours (1,732,544 S1)")
        print(f"RAM: {ram:.2f} GB | CPU: {cpu}%")

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'K Cap':<10} | {'Cand Rec':<10} | {'Avg Cand':<10} | {'F0.5':<8} | {'Prec':<8} | {'Rec':<8} | {'10k Time':<10} | {'Est Full Test':<14}")
    print("-" * 90)
    for label, r in benchmark_results.items():
        print(f"{str(r['K']):<10} | {r['candidate_recall']:<10} | {r['avg_candidates_per_s1']:<10} | {r['macro_f05']:<8} | {r['macro_precision']:<8} | {r['macro_recall']:<8} | {r['runtime_10k_s1_sec']:<10} | {r['est_full_test_hours']:<14}")

    benchmark_file = SCRATCH_DIR / "two_stage_retrieval_benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results to {benchmark_file}")

if __name__ == "__main__":
    run_benchmark()
