import os
import sys
import time
import re
import psutil
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

base_dir = Path(r"c:\Users\khush\OneDrive\Desktop\amzon ml challenge")
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
sys.path.append(str(src_dir))

from blocking import EntityBlocker

sub_amp = re.compile(r'&')
sub_punct = re.compile(r'[^\w\s]')
sub_space = re.compile(r'\s+')

def fast_norm_name(text):
    if not isinstance(text, str) or pd.isna(text): return ""
    return sub_space.sub(' ', sub_punct.sub(' ', sub_amp.sub(' and ', text.lower()))).strip()

def fast_norm_addr(text):
    if not isinstance(text, str) or pd.isna(text): return ""
    return sub_space.sub(' ', sub_punct.sub(' ', text.lower())).strip()

def extract_digits(text: str) -> set:
    return set(re.findall(r'\b\d+\b', text))

def extract_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n: return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

# Load model
model = joblib.load(base_dir / "models" / "lightgbm_model.joblib")

# Load S1 sample
s1_path = base_dir / "dataset" / "test" / "test_source1.tsv"
df_s1 = pd.read_csv(s1_path, sep="\t", nrows=5000, dtype=str)

# Pre-build index on S2+S3 sample for testing
s2_path = base_dir / "dataset" / "test" / "test_source2.tsv"
df_s2 = pd.read_csv(s2_path, sep="\t", nrows=50000, dtype=str)

blocker = EntityBlocker()
target_attr = {}
target_parsed = {}

for mid, rn, ra, c in zip(df_s2['entity_id'], df_s2['business_name'], df_s2['business_address'], df_s2['country'].fillna('UNKNOWN')):
    nn, na = fast_norm_name(rn), fast_norm_addr(ra)
    target_attr[mid] = f"{c}\t{nn}\t{na}"
    target_parsed[mid] = (
        str(c), nn, na,
        set(nn.split()), set(na.split()),
        extract_ngrams(nn, 3), extract_ngrams(na, 3),
        extract_digits(na)
    )
    blocker.add_target_record(mid, str(c), nn, na)

# Optimized candidate lookup directly using pre-normalized tokens
def get_candidates_fast(nn, na, c, n_toks, a_toks, ngs):
    cands = set()
    if len(n_toks) >= 2:
        cands.update(blocker.name_pref2_idx.get((" ".join(n_toks[:2]), c), set()))
    elif n_toks:
        cands.update(blocker.name_pref2_idx.get((n_toks[0], c), set()))

    if len(a_toks) >= 2:
        cands.update(blocker.addr_pref2_idx.get((" ".join(a_toks[:2]), c), set()))
    elif a_toks:
        cands.update(blocker.addr_pref2_idx.get((a_toks[0], c), set()))

    for tok in [t for t in n_toks if t not in blocker.stop_name_tokens and len(t) > 2][:2]:
        cands.update(blocker.name_rare_tok_idx.get((tok, c), set()))

    for tok in [t for t in a_toks if t not in blocker.stop_addr_tokens and (len(t) > 2 or t.isdigit())][:2]:
        cands.update(blocker.addr_rare_tok_idx.get((tok, c), set()))

    for ng in [g for g in ngs if g not in blocker.stop_ngrams][:1]:
        cands.update(blocker.name_ngram_idx.get((ng, c), set()))

    return cands

def extract_fast_features(a1, tid):
    c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = target_parsed[tid]
    
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
    is_s2 = 1.0 if tid.startswith('S2-') else 0.0

    return [
        n_exact, n_jaccard, n_char_jaccard, a_jaccard, a_char_jaccard,
        d_jaccard, c_match, len_diff_n, len_diff_a, float(n_overlap),
        float(a_overlap), missing_addr, is_s2
    ]

t0 = time.time()
pair_meta = []
s1_attr_map = {}

for s1_id, r_n, r_a, c in zip(df_s1['entity_id'], df_s1['business_name'], df_s1['business_address'], df_s1['country'].fillna('UNKNOWN')):
    norm_n = fast_norm_name(r_n)
    norm_a = fast_norm_addr(r_a)
    n_toks = set(norm_n.split())
    a_toks = set(norm_a.split())
    ng_n = extract_ngrams(norm_n, 3)
    ng_a = extract_ngrams(norm_a, 3)
    d_a = extract_digits(norm_a)
    c_str = str(c)

    a1 = {
        'norm_n': norm_n, 'toks_n': n_toks, 'ngrams_n': ng_n,
        'norm_a': norm_a, 'toks_a': a_toks, 'ngrams_a': ng_a,
        'digits_a': d_a, 'c': c_str
    }
    s1_attr_map[s1_id] = a1

    cands = get_candidates_fast(norm_n, norm_a, c_str, list(n_toks), list(a_toks), list(ng_n))
    for tid in cands:
        if tid in target_parsed:
            pair_meta.append((s1_id, tid))

t_cand = time.time() - t0

X_batch = np.zeros((len(pair_meta), 13), dtype=np.float32)
for idx, (s1_id, tid) in enumerate(pair_meta):
    X_batch[idx] = extract_fast_features(s1_attr_map[s1_id], tid)

t_feat = time.time() - t0 - t_cand

probs = model.predict_proba(X_batch)[:, 1]
t_pred = time.time() - t0 - t_cand - t_feat

t_total = time.time() - t0
print(f"5000 S1 FULL (Candidates: {len(pair_meta):,}): Total {t_total:.3f}s ({5000/t_total:.1f} S1/sec)")
print(f"Breakdown: Cand Gen: {t_cand:.3f}s, Feat Ext: {t_feat:.3f}s, Model Pred: {t_pred:.3f}s")
