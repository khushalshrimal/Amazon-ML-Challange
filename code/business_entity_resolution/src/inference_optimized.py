"""
Optimized Inference Module for Amazon ML Challenge Business Entity Resolution.
Memory-safe & ultra-fast on-demand target feature caching.
Preserves exact inference semantics, Strategy 3 blocker, 13 features, LightGBM model, and threshold 0.98.
"""

import os
import sys
import time
import re
import gc
import joblib
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from blocking import EntityBlocker

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

class OnDemandTargetStore:
    """
    Memory-Safe Target Attribute Store with On-Demand Feature Caching.
    Raw target records are stored as compact string tuples.
    Target candidate sets, tokens, n-grams, and digit sets are parsed ONCE on first access and cached.
    """
    def __init__(self):
        self.raw_attr = {}  # mid -> (country, norm_n, norm_a)
        self.cache = {}     # mid -> parsed target tuple

    def add_target_raw(self, mid: str, country: str, norm_n: str, norm_a: str):
        self.raw_attr[mid] = (country, norm_n, norm_a)

    def get_parsed(self, mid: str):
        if mid not in self.cache:
            c, norm_n, norm_a = self.raw_attr[mid]
            toks_n = set(norm_n.split())
            toks_a = set(norm_a.split())
            ngrams_n = extract_ngrams(norm_n, 3)
            ngrams_a = extract_ngrams(norm_a, 3)
            digits_a = extract_digits(norm_a)

            self.cache[mid] = (
                c,                      # 0
                norm_n,                 # 1
                norm_a,                 # 2
                toks_n,                 # 3
                toks_a,                 # 4
                ngrams_n,               # 5
                ngrams_a,               # 6
                digits_a,               # 7
                len(norm_n),            # 8
                len(norm_a),            # 9
                len(toks_n),            # 10
                len(toks_a),            # 11
                len(ngrams_n),          # 12
                len(ngrams_a),          # 13
                len(digits_a),          # 14
                1.0 if mid.startswith('S2-') else 0.0 # 15
            )
        return self.cache[mid]

    def __len__(self):
        return len(self.raw_attr)

def precompute_s1_record(s1_id: str, country: str, raw_n: str, raw_a: str) -> dict:
    """Precompute reusable S1 attributes once before candidate processing."""
    norm_n = fast_norm_name(raw_n)
    norm_a = fast_norm_addr(raw_a)
    toks_n = set(norm_n.split())
    toks_a = set(norm_a.split())
    ngrams_n = extract_ngrams(norm_n, 3)
    ngrams_a = extract_ngrams(norm_a, 3)
    digits_a = extract_digits(norm_a)

    return {
        's1_id': s1_id,
        'c': country,
        'norm_n': norm_n,
        'norm_a': norm_a,
        'toks_n': toks_n,
        'toks_a': toks_a,
        'ngrams_n': ngrams_n,
        'ngrams_a': ngrams_a,
        'digits_a': digits_a,
        'len_n': len(norm_n),
        'len_a': len(norm_a),
        'len_toks_n': len(toks_n),
        'len_toks_a': len(toks_a),
        'len_ng_n': len(ngrams_n),
        'len_ng_a': len(ngrams_a),
        'len_dig_a': len(digits_a),
        'raw_rec': {
            'entity_id': s1_id,
            'business_name': norm_n,
            'business_address': norm_a,
            'country': country
        }
    }

def fill_features_optimized(X: np.ndarray, row_idx: int, s1: dict, target_store: OnDemandTargetStore, tid: str):
    """
    Computes exact 13 features directly into pre-allocated NumPy array X[row_idx, :].
    Uses on-demand parsed target cache, set intersection and length arithmetic (len(A|B) = len(A)+len(B)-len(A&B))
    to avoid set union creations and Python float list allocations.
    """
    t_parsed = target_store.get_parsed(tid)
    
    c2 = t_parsed[0]
    norm_n2 = t_parsed[1]
    norm_a2 = t_parsed[2]
    toks_n2 = t_parsed[3]
    toks_a2 = t_parsed[4]
    ngrams_n2 = t_parsed[5]
    ngrams_a2 = t_parsed[6]
    digits_a2 = t_parsed[7]

    len_n2 = t_parsed[8]
    len_a2 = t_parsed[9]
    len_tn2 = t_parsed[10]
    len_ta2 = t_parsed[11]
    len_ng2_n = t_parsed[12]
    len_ng2_a = t_parsed[13]
    len_d2 = t_parsed[14]
    is_s2 = t_parsed[15]

    # Feature 0: Name exact match
    n_exact = 1.0 if s1['norm_n'] == norm_n2 and s1['norm_n'] != "" else 0.0

    # Feature 1 & 9: Name token Jaccard and overlap
    len_tn1 = s1['len_toks_n']
    if len_tn1 > 0 and len_tn2 > 0:
        n_overlap = len(s1['toks_n'] & toks_n2)
        n_jaccard = n_overlap / (len_tn1 + len_tn2 - n_overlap)
    else:
        n_overlap = 0
        n_jaccard = 0.0

    # Feature 2: Name 3-gram char Jaccard
    len_ng1_n = s1['len_ng_n']
    if len_ng1_n > 0 and len_ng2_n > 0:
        ng_n_overlap = len(s1['ngrams_n'] & ngrams_n2)
        n_char_jaccard = ng_n_overlap / (len_ng1_n + len_ng2_n - ng_n_overlap)
    else:
        n_char_jaccard = 0.0

    # Feature 3 & 10: Address token Jaccard and overlap
    len_ta1 = s1['len_toks_a']
    if len_ta1 > 0 and len_ta2 > 0:
        a_overlap = len(s1['toks_a'] & toks_a2)
        a_jaccard = a_overlap / (len_ta1 + len_ta2 - a_overlap)
    else:
        a_overlap = 0
        a_jaccard = 0.0

    # Feature 4: Address 3-gram char Jaccard
    len_ng1_a = s1['len_ng_a']
    if len_ng1_a > 0 and len_ng2_a > 0:
        ng_a_overlap = len(s1['ngrams_a'] & ngrams_a2)
        a_char_jaccard = ng_a_overlap / (len_ng1_a + len_ng2_a - ng_a_overlap)
    else:
        a_char_jaccard = 0.0

    # Feature 5: Address digit Jaccard
    len_d1 = s1['len_dig_a']
    if len_d1 > 0 and len_d2 > 0:
        d_overlap = len(s1['digits_a'] & digits_a2)
        d_jaccard = d_overlap / (len_d1 + len_d2 - d_overlap)
    else:
        d_jaccard = 0.5

    # Feature 6: Country exact match
    c_match = 1.0 if s1['c'] == c2 and s1['c'] != "" else 0.0

    # Feature 7 & 8: Length differences
    len_diff_n = float(abs(s1['len_n'] - len_n2))
    len_diff_a = float(abs(s1['len_a'] - len_a2))

    # Feature 11: Missing address flag
    missing_addr = 1.0 if s1['len_a'] == 0 or len_a2 == 0 else 0.0

    # Populate array directly
    X[row_idx, 0] = n_exact
    X[row_idx, 1] = n_jaccard
    X[row_idx, 2] = n_char_jaccard
    X[row_idx, 3] = a_jaccard
    X[row_idx, 4] = a_char_jaccard
    X[row_idx, 5] = d_jaccard
    X[row_idx, 6] = c_match
    X[row_idx, 7] = len_diff_n
    X[row_idx, 8] = len_diff_a
    X[row_idx, 9] = float(n_overlap)
    X[row_idx, 10] = float(a_overlap)
    X[row_idx, 11] = missing_addr
    X[row_idx, 12] = is_s2
