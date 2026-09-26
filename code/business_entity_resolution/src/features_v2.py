"""
Expanded Pairwise Feature Engineering Module (v2) for Submission 2.

Features included:
1. Name: Exact, Token Jaccard, Overlap, RapidFuzz Ratio, Partial Ratio, WRatio, Token Set Ratio, 3gram/4gram Jaccard, Prefix3/4, Len Diff & Ratio.
2. Address: Exact, Token Jaccard, Overlap, RapidFuzz Ratio, Token Set Ratio, 3gram Jaccard, Digit Jaccard, House Number Match, Pincode Match, Len Diff.
3. Phone & Email: Phone exact extract, Email domain match extract.
4. Country & Source: Country match, S2 flag.
5. Cross-field: Name x Country, Addr x Country, Name x Addr, Blocker score.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    from .normalize import normalize_name, normalize_address, normalize_country
except ImportError:
    from normalize import normalize_name, normalize_address, normalize_country

DIGIT_REGEX = re.compile(r'\b\d+\b')
PINCODE_REGEX = re.compile(r'\b\d{5,6}\b')
PHONE_REGEX = re.compile(r'\b\d{10,12}\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')

def extract_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    if inter == 0:
        return 0.0
    return inter / len(set1 | set2)

def extract_house_number(addr: str) -> str:
    m = DIGIT_REGEX.findall(addr)
    return m[0] if m else ""

def extract_pincode(addr: str) -> str:
    m = PINCODE_REGEX.findall(addr)
    return m[0] if m else ""

def extract_phone(text: str) -> str:
    m = PHONE_REGEX.findall(text)
    return m[0] if m else ""

def extract_email_domain(text: str) -> str:
    m = EMAIL_REGEX.findall(text)
    return m[0].lower() if m else ""

class OnDemandTargetStoreV2:
    """
    Enhanced Target Store with pre-extracted structured attributes & RapidFuzz caching.
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
            ngrams_n3 = extract_ngrams(norm_n, 3)
            ngrams_n4 = extract_ngrams(norm_n, 4)
            ngrams_a3 = extract_ngrams(norm_a, 3)
            digits_a = set(DIGIT_REGEX.findall(norm_a))

            house_num = extract_house_number(norm_a)
            pincode = extract_pincode(norm_a)
            phone = extract_phone(norm_n + " " + norm_a)
            email_dom = extract_email_domain(norm_n + " " + norm_a)

            self.cache[mid] = {
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
        return self.cache[mid]

def precompute_s1_v2(s1_id: str, country: str, raw_n: str, raw_a: str) -> dict:
    norm_n = normalize_name(raw_n)
    norm_a = normalize_address(raw_a)
    c = normalize_country(country)

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

    return {
        's1_id': s1_id,
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
        'len_a': len(norm_a)
    }

FEATURE_NAMES_V2 = [
    'name_exact_match', 'name_jaccard', 'name_token_overlap',
    'name_rapidfuzz_ratio', 'name_partial_ratio', 'name_wratio', 'name_token_set_sim',
    'name_char_3gram_jaccard', 'name_char_4gram_jaccard', 'name_prefix3_match', 'name_prefix4_match',
    'name_len_diff', 'name_len_ratio',
    'addr_exact_match', 'addr_jaccard', 'addr_token_overlap',
    'addr_rapidfuzz_ratio', 'addr_token_set_sim', 'addr_char_3gram_jaccard',
    'addr_digit_jaccard', 'addr_house_num_match', 'addr_pincode_match', 'addr_len_diff',
    'phone_exact_match', 'email_domain_match',
    'country_exact_match', 'is_s2',
    'name_x_country', 'addr_x_country', 'name_x_addr', 'name_jaccard_x_addr_jaccard', 'blocker_score'
]

def fill_features_v2(X: np.ndarray, row_idx: int, s1: dict, target_store: OnDemandTargetStoreV2, tid: str, blocker_score: float = 1.0):
    t = target_store.get_parsed(tid)

    # 1. Name Features
    n1, n2 = s1['norm_n'], t['norm_n']
    name_exact = 1.0 if n1 and n1 == n2 else 0.0
    name_jaccard = jaccard_similarity(s1['toks_n'], t['toks_n'])
    name_overlap = float(len(s1['toks_n'] & t['toks_n']))

    if name_jaccard == 0.0 and not name_exact and s1['norm_n'][:3] != t['norm_n'][:3]:
        rf_ratio = 0.0
        part_ratio = 0.0
        w_ratio = 0.0
        tok_set_ratio = 0.0
    else:
        rf_ratio = fuzz.ratio(n1, n2) / 100.0
        part_ratio = fuzz.partial_ratio(n1, n2) / 100.0
        w_ratio = fuzz.WRatio(n1, n2) / 100.0
        tok_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0

    n3_jaccard = jaccard_similarity(s1['ngrams_n3'], t['ngrams_n3'])
    n4_jaccard = jaccard_similarity(s1['ngrams_n4'], t['ngrams_n4'])

    pref3_match = 1.0 if n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3 else 0.0
    pref4_match = 1.0 if n1[:4] == n2[:4] and len(n1) >= 4 and len(n2) >= 4 else 0.0

    len_diff_n = float(abs(s1['len_n'] - t['len_n']))
    max_len_n = max(s1['len_n'], t['len_n'])
    len_ratio_n = (min(s1['len_n'], t['len_n']) / max_len_n) if max_len_n > 0 else 1.0

    # 2. Address Features
    a1, a2 = s1['norm_a'], t['norm_a']
    addr_exact = 1.0 if a1 and a1 == a2 else 0.0
    addr_jaccard = jaccard_similarity(s1['toks_a'], t['toks_a'])
    addr_overlap = float(len(s1['toks_a'] & t['toks_a']))

    if addr_jaccard == 0.0 and not addr_exact:
        addr_rf_ratio = 0.0
        addr_tok_set_ratio = 0.0
    else:
        addr_rf_ratio = fuzz.ratio(a1, a2) / 100.0
        addr_tok_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0

    a3_jaccard = jaccard_similarity(s1['ngrams_a3'], t['ngrams_a3'])
    digit_jaccard = jaccard_similarity(s1['digits_a'], t['digits_a'])

    if s1['house_num'] and t['house_num']:
        house_num_match = 1.0 if s1['house_num'] == t['house_num'] else 0.0
    else:
        house_num_match = 0.5

    if s1['pincode'] and t['pincode']:
        pincode_match = 1.0 if s1['pincode'] == t['pincode'] else 0.0
    else:
        pincode_match = 0.5

    len_diff_a = float(abs(s1['len_a'] - t['len_a']))

    # 3. Phone & Email Features
    if s1['phone'] and t['phone']:
        phone_match = 1.0 if s1['phone'] == t['phone'] else 0.0
    else:
        phone_match = 0.5

    if s1['email_dom'] and t['email_dom']:
        email_dom_match = 1.0 if s1['email_dom'] == t['email_dom'] else 0.0
    else:
        email_dom_match = 0.5

    # 4. Country & Source Features
    c1, c2 = s1['c'], t['c']
    country_match = 1.0 if c1 and c1 == c2 else 0.0
    is_s2 = t['is_s2']

    # 5. Cross-field Interactions
    name_x_c = rf_ratio * country_match
    addr_x_c = addr_rf_ratio * country_match
    name_x_addr = rf_ratio * addr_rf_ratio
    jaccard_x = name_jaccard * addr_jaccard

    X[row_idx, :] = [
        name_exact, name_jaccard, name_overlap,
        rf_ratio, part_ratio, w_ratio, tok_set_ratio,
        n3_jaccard, n4_jaccard, pref3_match, pref4_match,
        len_diff_n, len_ratio_n,
        addr_exact, addr_jaccard, addr_overlap,
        addr_rf_ratio, addr_tok_tok_set_ratio if 'addr_tok_tok_set_ratio' in locals() else addr_tok_set_ratio,
        a3_jaccard, digit_jaccard, house_num_match, pincode_match, len_diff_a,
        phone_match, email_dom_match,
        country_match, is_s2,
        name_x_c, addr_x_c, name_x_addr, jaccard_x, float(blocker_score)
    ]
