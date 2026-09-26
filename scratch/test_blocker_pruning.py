"""
Experimental Script: Blocker Pruning Threshold Analysis
Goal: Empirically test safer pruning thresholds on the ground-truth validation dataset
WITHOUT modifying production files or changing Strategy 3 blocker semantics.

DO NOT MODIFY PRODUCTION FILES.
DO NOT CHANGE STRATEGY 3 KEY DEFINITIONS.
DO NOT CHANGE 13 FEATURES OR LIGHTGBM MODEL.
"""

import os
import sys
import time
import gc
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Path setup
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address
from blocking import EntityBlocker, extract_3grams

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 80, flush=True)
print("=== AMAZON ML CHALLENGE — BLOCKER PRUNING THRESHOLD EXPERIMENT ===", flush=True)
print("=" * 80, flush=True)
print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

# 1. LOAD 5,000 GROUND-TRUTH VALIDATION S1 ENTITIES & TRUE MATCH PAIRS
train_dir = base_dir / "dataset" / "train"
gt_path = train_dir / "train_ground_truth.tsv"

print("Loading 5,000 Ground-Truth S1 Validation Entities...", flush=True)
gt_df = pd.read_csv(gt_path, sep="\t", nrows=5000, dtype=str)

val_gt_map = {}
val_gt_pairs = set()
s1_needed = set()
s2_needed = set()
s3_needed = set()

for _, r in gt_df.iterrows():
    s1_id = r['source1_entity_id']
    s1_needed.add(s1_id)
    m_str = str(r['matched_entity_ids']).strip() if pd.notna(r['matched_entity_ids']) else ""
    if m_str:
        m_set = set(m.strip() for m in m_str.split(',') if m.strip())
    else:
        m_set = set()
    val_gt_map[s1_id] = m_set
    for m in m_set:
        val_gt_pairs.add((s1_id, m))
        if m.startswith("S2-"):
            s2_needed.add(m)
        elif m.startswith("S3-"):
            s3_needed.add(m)

print(f"Loaded {len(val_gt_map):,} GT S1 entities, {len(val_gt_pairs):,} true match pairs.", flush=True)

# Load S1 records
s1_records = []
for chunk in pd.read_csv(train_dir / "train_source1.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s1_needed)]
    s1_records.append(m)
    if len(pd.concat(s1_records)) >= len(s1_needed):
        break

s1_val_df = pd.concat(s1_records, ignore_index=True)
s1_val_dict = s1_val_df.set_index('entity_id').to_dict(orient='index')

# Load target pool records
print("Loading target pool records (GT targets + background sample)...", flush=True)
s2_records = []
for chunk in pd.read_csv(train_dir / "train_source2.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s2_needed)]
    s2_records.append(m)
    if len(pd.concat(s2_records)) >= len(s2_needed):
        break

s3_records = []
for chunk in pd.read_csv(train_dir / "train_source3.tsv", sep="\t", chunksize=200000, dtype=str):
    m = chunk[chunk['entity_id'].isin(s3_needed)]
    s3_records.append(m)
    if len(pd.concat(s3_records)) >= len(s3_needed):
        break

s2_bg = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=150000, dtype=str)
s3_bg = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=150000, dtype=str)

target_pool = pd.concat([pd.concat(s2_records), pd.concat(s3_records), s2_bg, s3_bg], ignore_index=True).drop_duplicates(subset=['entity_id'])
target_recs = target_pool.to_dict('records')
print(f"Target pool contains {len(target_recs):,} unique target records. RAM: {get_ram_mb():.1f} MB\n", flush=True)

# Pre-normalize S1 query records
s1_queries = []
for s1_id, r in s1_val_dict.items():
    c = str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    if not c or c != c: c = 'UNKNOWN'
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    nn = normalize_name(raw_n)
    na = normalize_address(raw_a)
    s1_queries.append({
        'entity_id': s1_id,
        'country': c,
        'nn': nn,
        'na': na,
        'raw_rec': r
    })

# 2. DEFINE EXPERIMENTAL BLOCKER WITH VARIABLE PRUNING THRESHOLD
class PrunableEntityBlocker:
    """
    Experimental Blocker allowing uniform dynamic max_posting_limit pruning
    across all Strategy 3 Multi-Key Union channels.
    """
    def __init__(self, max_posting_limit: int):
        self.max_posting_limit = max_posting_limit
        self.name_pref2_idx = defaultdict(set)
        self.addr_pref2_idx = defaultdict(set)
        self.name_rare_tok_idx = defaultdict(set)
        self.addr_rare_tok_idx = defaultdict(set)
        self.name_ngram_idx = defaultdict(set)

        self.stop_name_tokens = set()
        self.stop_addr_tokens = set()
        self.stop_ngrams = set()
        self.pruned_keys_count = 0

    def build_index(self, target_records: list):
        self.name_pref2_idx.clear()
        self.addr_pref2_idx.clear()
        self.name_rare_tok_idx.clear()
        self.addr_rare_tok_idx.clear()
        self.name_ngram_idx.clear()
        self.stop_name_tokens.clear()
        self.stop_addr_tokens.clear()
        self.stop_ngrams.clear()

        # Step 1: Count frequency
        name_token_freq = Counter()
        addr_token_freq = Counter()
        name_ngram_freq = Counter()

        normalized_targets = []
        for rec in target_records:
            mid = rec['entity_id']
            c = rec.get('country', 'UNKNOWN')
            if not c or c != c: c = 'UNKNOWN'
            rn = str(rec.get('business_name', '')) if rec.get('business_name') is not None else ""
            ra = str(rec.get('business_address', '')) if rec.get('business_address') is not None else ""
            nn = normalize_name(rn)
            na = normalize_address(ra)
            n_toks = nn.split()
            a_toks = na.split()
            ngs = extract_3grams(nn)

            name_token_freq.update(n_toks)
            addr_token_freq.update(a_toks)
            name_ngram_freq.update(ngs)

            normalized_targets.append((mid, c, nn, na, n_toks, a_toks, ngs))

        max_n_freq = min(len(target_records) * 0.03, self.max_posting_limit)
        max_a_freq = min(len(target_records) * 0.01, self.max_posting_limit)
        max_ng_freq = min(len(target_records) * 0.05, self.max_posting_limit)

        self.stop_name_tokens = {t for t, cnt in name_token_freq.items() if cnt > max_n_freq}
        self.stop_addr_tokens = {t for t, cnt in addr_token_freq.items() if cnt > max_a_freq}
        self.stop_ngrams = {ng for ng, cnt in name_ngram_freq.items() if cnt > max_ng_freq}

        # Step 2: Build posting lists
        for mid, c, nn, na, n_toks, a_toks, ngs in normalized_targets:
            if len(n_toks) >= 2:
                self.name_pref2_idx[(" ".join(n_toks[:2]), c)].add(mid)
            elif n_toks:
                self.name_pref2_idx[(n_toks[0], c)].add(mid)

            if len(a_toks) >= 2:
                self.addr_pref2_idx[(" ".join(a_toks[:2]), c)].add(mid)
            elif a_toks:
                self.addr_pref2_idx[(a_toks[0], c)].add(mid)

            for tok in [t for t in n_toks if t not in self.stop_name_tokens and len(t) > 2][:2]:
                self.name_rare_tok_idx[(tok, c)].add(mid)

            for tok in [t for t in a_toks if t not in self.stop_addr_tokens and (len(t) > 2 or t.isdigit())][:2]:
                self.addr_rare_tok_idx[(tok, c)].add(mid)

            for ng in [g for g in ngs if g not in self.stop_ngrams][:1]:
                self.name_ngram_idx[(ng, c)].add(mid)

        # Step 3: Prune posting lists across ALL channels exceeding max_posting_limit
        self.pruned_keys_count = 0
        channels = [
            self.name_pref2_idx,
            self.addr_pref2_idx,
            self.name_rare_tok_idx,
            self.addr_rare_tok_idx,
            self.name_ngram_idx
        ]
        for ch in channels:
            to_remove = [k for k, s in ch.items() if len(s) > self.max_posting_limit]
            for k in to_remove:
                del ch[k]
                self.pruned_keys_count += 1

    def get_candidates(self, s1_record: dict) -> set:
        c = s1_record.get('country', 'UNKNOWN')
        if not c or c != c: c = 'UNKNOWN'

        raw_n = str(s1_record.get('business_name', '')) if s1_record.get('business_name') is not None else ""
        raw_a = str(s1_record.get('business_address', '')) if s1_record.get('business_address') is not None else ""

        nn = normalize_name(raw_n)
        na = normalize_address(raw_a)

        cands = set()

        n_toks = nn.split()
        if len(n_toks) >= 2:
            cands.update(self.name_pref2_idx.get((" ".join(n_toks[:2]), c), set()))
        elif n_toks:
            cands.update(self.name_pref2_idx.get((n_toks[0], c), set()))

        a_toks = na.split()
        if len(a_toks) >= 2:
            cands.update(self.addr_pref2_idx.get((" ".join(a_toks[:2]), c), set()))
        elif a_toks:
            cands.update(self.addr_pref2_idx.get((a_toks[0], c), set()))

        for tok in [t for t in n_toks if t not in self.stop_name_tokens and len(t) > 2][:2]:
            cands.update(self.name_rare_tok_idx.get((tok, c), set()))

        for tok in [t for t in a_toks if t not in self.stop_addr_tokens and (len(t) > 2 or t.isdigit())][:2]:
            cands.update(self.addr_rare_tok_idx.get((tok, c), set()))

        ngs = extract_3grams(nn)
        for ng in [g for g in ngs if g not in self.stop_ngrams][:1]:
            cands.update(self.name_ngram_idx.get((ng, c), set()))

        return cands

# 3. RUN EXPERIMENTS ACROSS PRUNING THRESHOLDS
thresholds = [
    (100000, "100,000 (Current)"),
    (50000,  "50,000"),
    (25000,  "25,000"),
    (10000,  "10,000"),
    (5000,   "5,000"),
    (2500,   "2,500")
]

print("--- RUNNING BLOCKER PRUNING EXPERIMENT ON 5,000 GT S1 VALIDATION SET ---", flush=True)

results = []
baseline_recall = None

for limit, label in thresholds:
    t0_exp = time.time()
    blocker_exp = PrunableEntityBlocker(max_posting_limit=limit)
    blocker_exp.build_index(target_recs)
    
    total_cands = 0
    captured_gt = 0
    cands_per_s1 = []
    
    for q in s1_queries:
        cands = blocker_exp.get_candidates(q['raw_rec'])
        cnt = len(cands)
        total_cands += cnt
        cands_per_s1.append(cnt)
        
        gt_set = val_gt_map.get(q['entity_id'], set())
        captured_gt += len(cands & gt_set)
        
    runtime = time.time() - t0_exp
    cur_ram = get_ram_mb()
    
    recall = (captured_gt / len(val_gt_pairs)) * 100.0 if val_gt_pairs else 0.0
    if baseline_recall is None:
        baseline_recall = recall
        recall_loss = 0.0
    else:
        recall_loss = baseline_recall - recall
        
    avg_cands = total_cands / len(s1_queries)
    max_cands = max(cands_per_s1)
    
    print(f"Limit [{label:17s}]: Recall = {recall:.2f}% (Loss = {recall_loss:.2f}%) | Pairs = {total_cands:9,d} | Avg/S1 = {avg_cands:7.1f} | Max/S1 = {max_cands:6,d} | Pruned Keys = {blocker_exp.pruned_keys_count:4d} | Time = {runtime:.2f}s", flush=True)
    
    results.append({
        'Limit': label,
        'Limit_Int': limit,
        'Candidate Pairs': total_cands,
        'Recall': recall,
        'Recall Loss': recall_loss,
        'Avg Cand/S1': avg_cands,
        'Max Cand/S1': max_cands,
        'Pruned Keys': blocker_exp.pruned_keys_count,
        'Runtime': runtime,
        'Peak RAM': cur_ram
    })
    
    del blocker_exp
    gc.collect()

# 4. PRINT FORMATTED REPORT
print("\n" + "=" * 90, flush=True)
print("=== BLOCKER PRUNING EXPERIMENT ===", flush=True)
print("=" * 90, flush=True)
print(f"Validation Ground-Truth Set: {len(val_gt_map):,} S1 Entities | {len(val_gt_pairs):,} True Match Pairs\n", flush=True)

print("| Limit | Candidate Pairs | Recall | Recall Loss | Avg Cand/S1 | Max Cand/S1 | Pruned Keys | Runtime | Peak RAM |", flush=True)
print("|---|---|---|---|---|---|---|---|---|", flush=True)
for r in results:
    print(f"| {r['Limit']} | {r['Candidate Pairs']:,} | {r['Recall']:.2f}% | {r['Recall Loss']:.2f}% | {r['Avg Cand/S1']:.1f} | {r['Max Cand/S1']:,} | {r['Pruned Keys']:,} | {r['Runtime']:.2f} s | {r['Peak RAM']:.1f} MB |", flush=True)

print("\n--- FINDINGS SUMMARY ---", flush=True)
print(f"1. Baseline (Current 100,000 Limit): {results[0]['Recall']:.2f}% Recall, {results[0]['Candidate Pairs']:,} candidate pairs ({results[0]['Avg Cand/S1']:.1f} candidates/S1).", flush=True)

best_tradeoff = None
for r in results[1:]:
    if r['Recall Loss'] <= 0.20:
        best_tradeoff = r

if best_tradeoff:
    reduction = (1.0 - (best_tradeoff['Candidate Pairs'] / results[0]['Candidate Pairs'])) * 100.0
    print(f"2. Preserved Recall Threshold: Limit {best_tradeoff['Limit']} preserves {best_tradeoff['Recall']:.2f}% recall (only {best_tradeoff['Recall Loss']:.2f}% loss) while reducing candidates by {reduction:.1f}% to {best_tradeoff['Avg Cand/S1']:.1f} candidates/S1.", flush=True)
else:
    print("2. Preserved Recall Threshold: Inspect metrics table above for trade-offs.", flush=True)

print("=" * 90 + "\n", flush=True)

print("To re-run this experimental script at any time, execute:", flush=True)
print(r"py -3 scratch\test_blocker_pruning.py", flush=True)
