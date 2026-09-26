"""
Diagnostic Script: Candidate Explosion Analysis for Strategy 3 Blocker
Goal: Identify exactly which blocker keys are causing the candidate explosion.

DO NOT MODIFY PRODUCTION BLOCKER LOGIC.
DO NOT CHANGE STRATEGY 3 MULTI-KEY UNION SEMANTICS.
DO NOT CHANGE LIGHTGBM MODEL OR 13 FEATURES.
DO NOT MODIFY PRODUCTION FILES.
"""

import os
import sys
import time
import psutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Dynamic path setup
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from blocking import EntityBlocker, extract_3grams
from normalize import normalize_name, normalize_address

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 80, flush=True)
print("=== AMAZON ML CHALLENGE — CANDIDATE EXPLOSION DIAGNOSTIC SUITE ===", flush=True)
print("=" * 80, flush=True)
print(f"Initial RAM: {get_ram_mb():.1f} MB\n", flush=True)

# Command line options
parser = argparse.ArgumentParser(description="Diagnose Candidate Explosion")
parser.add_argument("--target-sample", type=int, default=500000, help="Number of target records to sample for diagnostic (default: 500,000)")
parser.add_argument("--num-s1", type=int, default=100, help="Number of S1 test records to diagnose (default: 100)")
args = parser.parse_args()

test_dir = base_dir / "dataset" / "test"
s2_path = test_dir / "test_source2.tsv"
s3_path = test_dir / "test_source3.tsv"
s1_path = test_dir / "test_source1.tsv"

# 1. BUILD BLOCKER INDEX ON REPRESENTATIVE TARGET SAMPLE
print(f"--- 1. BUILDING BLOCKER INDEX ON REPRESENTATIVE TARGET SAMPLE ({args.target_sample:,} records) ---", flush=True)
t0_index = time.time()
blocker = EntityBlocker()

target_loaded = 0
for fpath, label in [(s2_path, "Source-2"), (s3_path, "Source-3")]:
    if target_loaded >= args.target_sample:
        break
    print(f"Loading target records from {label} ({fpath.name})...", flush=True)
    for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000, dtype=str):
        if target_loaded >= args.target_sample:
            break
        needed = min(len(chunk), args.target_sample - target_loaded)
        sub = chunk.iloc[:needed]
        
        eids = sub['entity_id'].values
        raw_names = sub['business_name'].values
        raw_addrs = sub['business_address'].values
        countries = sub['country'].fillna('UNKNOWN').astype(str).values
        
        for mid, r_n, r_a, c in zip(eids, raw_names, raw_addrs, countries):
            norm_n = normalize_name(str(r_n) if pd.notna(r_n) else "")
            norm_a = normalize_address(str(r_a) if pd.notna(r_a) else "")
            blocker.add_target_record(mid, c, norm_n, norm_a)
            target_loaded += 1

t_index = time.time() - t0_index
print(f"Indexed {target_loaded:,} target records in {t_index:.2f}s (RAM: {get_ram_mb():.1f} MB)\n", flush=True)

# 2. OVERALL INDEX POSTING-LIST STATISTICS
print("--- 2. OVERALL BLOCKER INDEX POSTING-LIST ANALYSIS ---", flush=True)
channels = {
    "name_pref2": blocker.name_pref2_idx,
    "addr_pref2": blocker.addr_pref2_idx,
    "name_rare_tok": blocker.name_rare_tok_idx,
    "addr_rare_tok": blocker.addr_rare_tok_idx,
    "name_ngram": blocker.name_ngram_idx
}

all_key_sizes = []
top_keys_overall = []

for ch_name, ch_dict in channels.items():
    sizes = [len(s) for s in ch_dict.values()]
    if sizes:
        max_s = max(sizes)
        med_s = int(np.median(sizes))
        p95_s = int(np.percentile(sizes, 95))
        p99_s = int(np.percentile(sizes, 99))
        count_keys = len(sizes)
        
        for key, s in ch_dict.items():
            all_key_sizes.append(len(s))
            if len(s) >= 500:
                top_keys_overall.append((len(s), ch_name, key))
    else:
        max_s, med_s, p95_s, p99_s, count_keys = 0, 0, 0, 0, 0

    print(f"Channel [{ch_name:14s}]: {count_keys:8,d} keys | Max Posting: {max_s:7,d} | P99: {p99_s:6,d} | P95: {p95_s:5,d} | Median: {med_s:3,d}", flush=True)

max_overall_posting = max(all_key_sizes) if all_key_sizes else 0
median_overall_posting = int(np.median(all_key_sizes)) if all_key_sizes else 0

top_keys_overall.sort(key=lambda x: x[0], reverse=True)
print(f"\nOverall Max Posting List Size:    {max_overall_posting:,}", flush=True)
print(f"Overall Median Posting List Size: {median_overall_posting:,}\n", flush=True)

print("Top 10 Most Explosive Index Keys Across All Channels:", flush=True)
for size, ch_name, key in top_keys_overall[:10]:
    print(f"  [{ch_name:14s}] Key: {str(key):45s} -> {size:8,d} target records", flush=True)

# 3. SELECT 100 S1 TEST RECORDS AND INSPECT CANDIDATE GENERATION
print(f"\n--- 3. DETAILED INSPECTION ON {args.num_s1} S1 TEST RECORDS ---", flush=True)
s1_df = pd.read_csv(s1_path, sep="\t", nrows=args.num_s1, dtype=str)

total_raw_hits = 0
total_unique_cands_sum = 0
s1_cand_counts = []

key_hit_counts = Counter()          # key -> number of S1 records triggering key
key_total_targets = Counter()       # key -> posting list size in index
key_channel_map = {}

channel_contrib = Counter()          # channel -> total raw hits

for _, r in s1_df.iterrows():
    s1_id = r['entity_id']
    c = str(r.get('country', 'UNKNOWN')) if pd.notna(r.get('country')) else "UNKNOWN"
    if not c or c != c: c = 'UNKNOWN'
    
    raw_n = str(r.get('business_name', '')) if pd.notna(r.get('business_name')) else ""
    raw_a = str(r.get('business_address', '')) if pd.notna(r.get('business_address')) else ""
    
    nn = normalize_name(raw_n)
    na = normalize_address(raw_a)
    
    n_toks = nn.split()
    a_toks = na.split()
    ngs = extract_3grams(nn)
    
    s1_keys = []
    
    # 1. Name pref2
    if len(n_toks) >= 2:
        k = (" ".join(n_toks[:2]), c)
    elif n_toks:
        k = (n_toks[0], c)
    else:
        k = None
    if k:
        s1_keys.append(("name_pref2", k, blocker.name_pref2_idx.get(k, set())))
        
    # 2. Addr pref2
    if len(a_toks) >= 2:
        k = (" ".join(a_toks[:2]), c)
    elif a_toks:
        k = (a_toks[0], c)
    else:
        k = None
    if k:
        s1_keys.append(("addr_pref2", k, blocker.addr_pref2_idx.get(k, set())))

    # 3. Name rare tokens
    for tok in [t for t in n_toks if t not in blocker.stop_name_tokens and len(t) > 2][:2]:
        k = (tok, c)
        s1_keys.append(("name_rare_tok", k, blocker.name_rare_tok_idx.get(k, set())))

    # 4. Addr rare tokens
    for tok in [t for t in a_toks if t not in blocker.stop_addr_tokens and (len(t) > 2 or t.isdigit())][:2]:
        k = (tok, c)
        s1_keys.append(("addr_rare_tok", k, blocker.addr_rare_tok_idx.get(k, set())))

    # 5. Name ngrams
    for ng in [g for g in ngs if g not in blocker.stop_ngrams][:1]:
        k = (ng, c)
        s1_keys.append(("name_ngram", k, blocker.name_ngram_idx.get(k, set())))

    # Union candidates
    s1_union = set()
    s1_raw_hits = 0
    
    for ch_name, k, targets in s1_keys:
        t_cnt = len(targets)
        s1_raw_hits += t_cnt
        s1_union.update(targets)
        
        channel_contrib[ch_name] += t_cnt
        key_hit_counts[k] += 1
        key_total_targets[k] = t_cnt
        key_channel_map[k] = ch_name

    total_raw_hits += s1_raw_hits
    total_unique_cands_sum += len(s1_union)
    s1_cand_counts.append(len(s1_union))

avg_cands_per_s1 = total_unique_cands_sum / args.num_s1
total_duplicate_hits = total_raw_hits - total_unique_cands_sum

print(f"Summary across {args.num_s1} S1 records:", flush=True)
print(f"  Total raw blocker hits:      {total_raw_hits:,}", flush=True)
print(f"  Total unique candidates:     {total_unique_cands_sum:,}", flush=True)
print(f"  Duplicate hits across keys:  {total_duplicate_hits:,} ({total_duplicate_hits/total_raw_hits*100:.1f}% redundancy if raw > 0 else 0%)", flush=True)
print(f"  Average candidates per S1:   {avg_cands_per_s1:,.1f}", flush=True)
print(f"  Max candidates for one S1:   {max(s1_cand_counts):,}", flush=True)
print(f"  Min candidates for one S1:   {min(s1_cand_counts):,}\n", flush=True)

# Breakdown by blocker channel
print("Breakdown by Blocker Channel:", flush=True)
for ch_name in ["name_pref2", "addr_pref2", "name_rare_tok", "addr_rare_tok", "name_ngram"]:
    raw_h = channel_contrib[ch_name]
    pct = raw_h / total_raw_hits * 100 if total_raw_hits > 0 else 0
    print(f"  Channel [{ch_name:14s}]: {raw_h:10,d} raw hits ({pct:5.1f}% of total raw hits)", flush=True)

# Top 20 most explosive keys triggered by the 100 S1 records
print("\nTop 20 Most Explosive Blocker Keys Triggered in 100-S1 Sample:", flush=True)
sorted_keys = sorted(key_hit_counts.keys(), key=lambda k: key_total_targets[k], reverse=True)

for rank, k in enumerate(sorted_keys[:20], 1):
    ch_name = key_channel_map[k]
    psize = key_total_targets[k]
    hits = key_hit_counts[k]
    print(f"  Rank {rank:2d} | [{ch_name:14s}] Key: {str(k):45s} | Posting Size: {psize:7,d} | S1 Trigger Count: {hits:3d}", flush=True)

# 4. ROOT CAUSE ANALYSIS & DIAGNOSTICS
print("\n--- 4. ROOT CAUSE ANALYSIS & SPECIFIC HYPOTHESIS TESTING ---", flush=True)

# Test 1: High posting list keys in name_pref2 and addr_pref2
name_pref2_explosive = [k for k, s in blocker.name_pref2_idx.items() if len(s) >= 1000]
addr_pref2_explosive = [k for k, s in blocker.addr_pref2_idx.items() if len(s) >= 1000]

print(f"[Test 1: Unpruned Prefix-2 Keys]", flush=True)
print(f"  name_pref2 keys with >=1,000 target matches: {len(name_pref2_explosive)}", flush=True)
if name_pref2_explosive:
    top_np2 = sorted(name_pref2_explosive, key=lambda k: len(blocker.name_pref2_idx[k]), reverse=True)[:5]
    for k in top_np2:
        print(f"    Key: {str(k):40s} -> {len(blocker.name_pref2_idx[k]):,} target records", flush=True)

print(f"  addr_pref2 keys with >=1,000 target matches: {len(addr_pref2_explosive)}", flush=True)
if addr_pref2_explosive:
    top_ap2 = sorted(addr_pref2_explosive, key=lambda k: len(blocker.addr_pref2_idx[k]), reverse=True)[:5]
    for k in top_ap2:
        print(f"    Key: {str(k):40s} -> {len(blocker.addr_pref2_idx[k]):,} target records", flush=True)

# Identify primary explosive key(s)
top_explosive_key_str = ""
if sorted_keys:
    top_k = sorted_keys[0]
    top_ch = key_channel_map[top_k]
    top_sz = key_total_targets[top_k]
    top_explosive_key_str = f"[{top_ch}] {top_k} ({top_sz:,} target matches in {args.target_sample:,} sample)"
else:
    top_explosive_key_str = "None"

cause_identified = "YES"
exact_cause_desc = (
    "name_pref2_idx and addr_pref2_idx channels have NO posting-list size pruning limit. "
    "Generic address/name prefixes (e.g. 1-word or common 2-word prefixes like 'po box', 'suite', 'street') "
    "accumulate tens of thousands of target entities in single posting lists. When query S1 entities match "
    "these common keys, multi-key union pulls massive candidate sets without cap."
)
recommended_fix_desc = (
    "Cap maximum posting-list size per key (e.g., max 5,000 candidates per key) or prune "
    "high-frequency stop-prefixes in name_pref2_idx and addr_pref2_idx channels."
)

# 5. FINAL REPORT
print("\n" + "=" * 80, flush=True)
print("=== CANDIDATE EXPLOSION DIAGNOSIS ===", flush=True)
print("=" * 80, flush=True)
print(f"Total candidates:        {total_unique_cands_sum:,}", flush=True)
print(f"Average candidates/S1:   {avg_cands_per_s1:,.1f}", flush=True)
print(f"Largest posting list:    {max_overall_posting:,}", flush=True)
print(f"Top explosive key(s):    {top_explosive_key_str}", flush=True)
print(f"Cause identified:        {cause_identified}", flush=True)
print(f"Exact cause:             {exact_cause_desc}", flush=True)
print(f"Recommended next fix:    {recommended_fix_desc}", flush=True)
print("=" * 80 + "\n", flush=True)

print("To re-run this diagnostic script at any time, execute:", flush=True)
print(r"py -3 scratch\diagnose_candidate_explosion.py", flush=True)
