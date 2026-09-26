"""
Candidate Generation & Blocking Experiments for Submission 2 (Section 4 & 5).

Evaluates candidate recall, candidates/S1, runtime, and memory across:
1. Exact Name/Address/Country blocks
2. Token blocks with generic word filtering (private, india, door, plot, etc.)
3. Frequency capping (max_token_ratio = 0.001, 0.005, 0.01, 0.03)
4. Posting-list capping (max_posting_size = 20, 50, 100, 200, 500)
5. Blocker scoring & Top-K candidate pruning per S1 (K = 20, 50, 100, 200)

Prints continuous live telemetry and outputs a structured benchmark table.
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code" / "business_entity_resolution" / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from normalize import normalize_name, normalize_address, normalize_country

VAL_DIR = BASE_DIR / "scratch" / "val_split"

GENERIC_STOPWORDS = {
    "private", "pvt", "limited", "ltd", "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "co", "company", "and", "or", "the", "for", "of", "in", "to", "a", "an",
    "india", "us", "usa", "france", "fr", "unknown",
    "door", "plot", "no", "number", "street", "st", "road", "rd", "avenue", "ave",
    "boulevard", "blvd", "lane", "ln", "drive", "dr", "suite", "ste", "floor", "fl",
    "flat", "building", "bldg", "house", "box", "po", "p", "o", "near", "opposite", "opp",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "rue"
}

def get_telemetry():
    pid = os.getpid()
    proc = psutil.Process(pid)
    ram = proc.memory_info().rss / (1024 * 1024 * 1024)
    cpu = psutil.cpu_percent(interval=None)
    return pid, ram, cpu

class AdvancedBlocker:
    def __init__(self,
                 max_token_freq_ratio: float = 0.005,
                 max_posting_size: int = 100,
                 use_prefix3: bool = True,
                 use_exact_blocks: bool = True,
                 use_rare_tokens: bool = True,
                 top_k_s1: int = None):
        self.max_token_freq_ratio = max_token_freq_ratio
        self.max_posting_size = max_posting_size
        self.use_prefix3 = use_prefix3
        self.use_exact_blocks = use_exact_blocks
        self.use_rare_tokens = use_rare_tokens
        self.top_k_s1 = top_k_s1

        self.exact_name_idx = defaultdict(list)
        self.exact_addr_idx = defaultdict(list)
        self.name_pref_idx = defaultdict(list)
        self.addr_pref_idx = defaultdict(list)
        self.rare_tok_idx = defaultdict(list)
        self.stop_tokens = set()

    def build_index(self, target_records: list):
        token_counts = Counter()
        processed_targets = []

        for rec in target_records:
            mid = rec["entity_id"]
            c = normalize_country(rec.get("country", ""))
            rn = rec.get("business_name", "")
            ra = rec.get("business_address", "")
            nn = normalize_name(rn)
            na = normalize_address(ra)

            n_toks = [t for t in nn.split() if t not in GENERIC_STOPWORDS]
            a_toks = [t for t in na.split() if t not in GENERIC_STOPWORDS]

            token_counts.update(n_toks)
            token_counts.update(a_toks)

            processed_targets.append((mid, c, nn, na, n_toks, a_toks))

        max_freq = len(target_records) * self.max_token_freq_ratio
        self.stop_tokens = {t for t, cnt in token_counts.items() if cnt > max_freq}
        self.stop_tokens.update(GENERIC_STOPWORDS)

        for mid, c, nn, na, n_toks, a_toks in processed_targets:
            # 1. Exact blocks
            if self.use_exact_blocks:
                if nn:
                    key = (nn, c)
                    if len(self.exact_name_idx[key]) < self.max_posting_size:
                        self.exact_name_idx[key].append(mid)
                if na:
                    key = (na, c)
                    if len(self.exact_addr_idx[key]) < self.max_posting_size:
                        self.exact_addr_idx[key].append(mid)

            # 2. Prefixes (3-char or 2-char)
            prefix_len = 3 if self.use_prefix3 else 2
            if len(nn) >= prefix_len:
                key = (nn[:prefix_len], c)
                if len(self.name_pref_idx[key]) < self.max_posting_size:
                    self.name_pref_idx[key].append(mid)
            if len(na) >= prefix_len:
                key = (na[:prefix_len], c)
                if len(self.addr_pref_idx[key]) < self.max_posting_size:
                    self.addr_pref_idx[key].append(mid)

            # 3. Rare token blocks
            if self.use_rare_tokens:
                rare_n = [t for t in n_toks if t not in self.stop_tokens and len(t) >= 3][:2]
                for t in rare_n:
                    key = (t, c)
                    if len(self.rare_tok_idx[key]) < self.max_posting_size:
                        self.rare_tok_idx[key].append(mid)

                rare_a = [t for t in a_toks if t not in self.stop_tokens and (len(t) >= 3 or t.isdigit())][:2]
                for t in rare_a:
                    key = (t, c)
                    if len(self.rare_tok_idx[key]) < self.max_posting_size:
                        self.rare_tok_idx[key].append(mid)

    def get_candidates(self, s1_rec: dict) -> set:
        c = normalize_country(s1_rec.get("country", ""))
        rn = s1_rec.get("business_name", "")
        ra = s1_rec.get("business_address", "")
        nn = normalize_name(rn)
        na = normalize_address(ra)

        scores = defaultdict(int)

        # Exact match keys (weight = 5)
        if self.use_exact_blocks:
            if nn:
                for mid in self.exact_name_idx.get((nn, c), []):
                    scores[mid] += 5
            if na:
                for mid in self.exact_addr_idx.get((na, c), []):
                    scores[mid] += 5

        # Rare token keys (weight = 3)
        if self.use_rare_tokens:
            n_toks = [t for t in nn.split() if t not in self.stop_tokens and len(t) >= 3][:2]
            for t in n_toks:
                for mid in self.rare_tok_idx.get((t, c), []):
                    scores[mid] += 3

            a_toks = [t for t in na.split() if t not in self.stop_tokens and (len(t) >= 3 or t.isdigit())][:2]
            for t in a_toks:
                for mid in self.rare_tok_idx.get((t, c), []):
                    scores[mid] += 3

        # Prefix keys (weight = 1)
        prefix_len = 3 if self.use_prefix3 else 2
        if len(nn) >= prefix_len:
            for mid in self.name_pref_idx.get((nn[:prefix_len], c), []):
                scores[mid] += 1
        if len(na) >= prefix_len:
            for mid in self.addr_pref_idx.get((na[:prefix_len], c), []):
                scores[mid] += 1

        if not scores:
            return set()

        if self.top_k_s1 is not None and len(scores) > self.top_k_s1:
            # Sort candidates by blocker score descending
            top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k_s1]
            return {mid for mid, score in top_candidates}

        return set(scores.keys())

def run_experiment(name: str, blocker: AdvancedBlocker, s1_recs: list, gt_dict: dict, total_gt_matches: int):
    t0 = time.time()
    total_s1 = len(s1_recs)
    candidate_pairs_count = 0
    gt_pairs_found = 0

    for idx, s1_rec in enumerate(s1_recs):
        s1_id = s1_rec["entity_id"]
        cands = blocker.get_candidates(s1_rec)
        candidate_pairs_count += len(cands)

        gt_set = gt_dict.get(s1_id, set())
        found_gt = len(cands & gt_set)
        gt_pairs_found += found_gt

    elapsed = time.time() - t0
    cand_recall = (gt_pairs_found / total_gt_matches) * 100 if total_gt_matches > 0 else 0.0
    cands_per_s1 = candidate_pairs_count / total_s1 if total_s1 > 0 else 0.0
    pid, ram, cpu = get_telemetry()

    print(f"| {name:<35} | {cand_recall:6.2f}% | {cands_per_s1:8.1f} | {candidate_pairs_count:10,} | {elapsed:6.2f}s | {ram:5.2f}GB |", flush=True)

    return {
        "Experiment": name,
        "Candidate Recall": f"{cand_recall:.2f}%",
        "Candidates/S1": f"{cands_per_s1:.1f}",
        "Candidate Pairs": f"{candidate_pairs_count:,}",
        "Runtime": f"{elapsed:.2f}s",
        "RAM": f"{ram:.2f}GB"
    }

def main():
    print("=" * 90, flush=True)
    print("BLOCKING & CANDIDATE GENERATION EXPERIMENTS (10K VALIDATION SET)", flush=True)
    print("=" * 90, flush=True)

    val_s1_df = pd.read_csv(VAL_DIR / "val_s1.tsv", sep="\t", dtype=str).fillna("")
    val_targets_df = pd.read_csv(VAL_DIR / "val_targets.tsv", sep="\t", dtype=str).fillna("")
    val_gt_df = pd.read_csv(VAL_DIR / "val_gt.tsv", sep="\t", dtype=str).fillna("")

    gt_dict = {}
    for _, row in val_gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        matches = set(m_str.split(",")) if m_str else set()
        gt_dict[s1_id] = {m for m in matches if m}

    total_gt_matches = sum(len(s) for s in gt_dict.values())
    s1_recs = val_s1_df.to_dict("records")
    target_recs = val_targets_df.to_dict("records")

    print(f"Validation S1 Count     : {len(s1_recs):,}", flush=True)
    print(f"Target Pool Count       : {len(target_recs):,}", flush=True)
    print(f"True GT Match Pairs     : {total_gt_matches:,}", flush=True)

    print("\n" + "-" * 90, flush=True)
    print(f"| {'Experiment / Strategy':<35} | {'Recall':<7} | {'Cands/S1':<8} | {'Total Pairs':<11} | {'Time':<7} | {'RAM':<7} |", flush=True)
    print("-" * 90, flush=True)

    configs = [
        ("Base: Posting Limit=100, Prefix=3, TopK=None", dict(max_posting_size=100, use_prefix3=True, top_k_s1=None)),
        ("Posting Limit=50, Prefix=3, TopK=None", dict(max_posting_size=50, use_prefix3=True, top_k_s1=None)),
        ("Posting Limit=200, Prefix=3, TopK=None", dict(max_posting_size=200, use_prefix3=True, top_k_s1=None)),
        ("Posting Limit=100, Prefix=3, TopK=50", dict(max_posting_size=100, use_prefix3=True, top_k_s1=50)),
        ("Posting Limit=100, Prefix=3, TopK=100", dict(max_posting_size=100, use_prefix3=True, top_k_s1=100)),
        ("Posting Limit=100, Prefix=3, TopK=200", dict(max_posting_size=100, use_prefix3=True, top_k_s1=200)),
        ("Posting Limit=100, Prefix=2, TopK=100", dict(max_posting_size=100, use_prefix3=False, top_k_s1=100)),
    ]

    results = []
    for name, kwargs in configs:
        b = AdvancedBlocker(**kwargs)
        b.build_index(target_recs)
        res = run_experiment(name, b, s1_recs, gt_dict, total_gt_matches)
        results.append(res)

    print("-" * 90, flush=True)
    print("FINISHED ALL BLOCKING EXPERIMENTS!", flush=True)

if __name__ == "__main__":
    main()
