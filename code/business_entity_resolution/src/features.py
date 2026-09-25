"""
Feature extraction module for candidate entity pairs.
Extracts string similarities, token overlap, length metrics, and domain-specific matching signals.
"""

import re
from difflib import SequenceMatcher
import pandas as pd

try:
    from .normalize import normalize_name, normalize_address
except ImportError:
    from normalize import normalize_name, normalize_address

def token_jaccard(tokens1: set, tokens2: set) -> float:
    """Compute token Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    if inter == 0:
        return 0.0
    return inter / len(tokens1 | tokens2)

def char_seq_ratio(str1: str, str2: str) -> float:
    """Compute SequenceMatcher similarity ratio between two normalized strings."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def extract_digit_set(text: str) -> set:
    """Extract all multi-digit numbers from a string (e.g. house numbers, zip codes)."""
    return set(re.findall(r'\b\d+\b', text))

def digit_jaccard(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of digit sequences in address/name strings."""
    d1 = extract_digit_set(text1)
    d2 = extract_digit_set(text2)
    if not d1 or not d2:
        return 0.5  # neutral if digits absent in one or both
    return len(d1 & d2) / len(d1 | d2)

def extract_pair_features_precomputed(
    norm_n1: str, toks_n1: set, norm_a1: str, toks_a1: set, c1: str,
    norm_n2: str, toks_n2: set, norm_a2: str, toks_a2: set, c2: str,
    target_id: str,
    skip_slow_sim: bool = False
) -> dict:
    """
    Extract features from pre-normalized strings and pre-tokenized sets for maximum performance.
    """
    n_jaccard = token_jaccard(toks_n1, toks_n2)
    a_jaccard = token_jaccard(toks_a1, toks_a2)

    n_overlap = len(toks_n1 & toks_n2)
    a_overlap = len(toks_a1 & toks_a2)

    n_pref3_match = 1.0 if norm_n1[:3] == norm_n2[:3] and len(norm_n1) >= 3 and len(norm_n2) >= 3 else 0.0

    if skip_slow_sim and n_jaccard == 0 and a_jaccard == 0 and n_pref3_match == 0:
        n_char_ratio = 0.0
        a_char_ratio = 0.0
    else:
        n_char_ratio = char_seq_ratio(norm_n1, norm_n2)
        a_char_ratio = char_seq_ratio(norm_a1, norm_a2)

    return {
        'name_jaccard': n_jaccard,
        'name_char_ratio': n_char_ratio,
        'addr_jaccard': a_jaccard,
        'addr_char_ratio': a_char_ratio,
        'name_len_diff': float(abs(len(norm_n1) - len(norm_n2))),
        'addr_len_diff': float(abs(len(norm_a1) - len(norm_a2))),
        'name_token_overlap': float(n_overlap),
        'addr_token_overlap': float(a_overlap),
        'name_prefix3_match': n_pref3_match,
        'digit_match_ratio': digit_jaccard(norm_a1, norm_a2),
        'country_exact_match': 1.0 if c1 == c2 and c1 != "" else 0.0,
        'is_s2': 1.0 if target_id.startswith('S2-') else 0.0,
    }

def extract_pair_features(s1_record: dict, candidate_record: dict) -> dict:
    """
    Convenience function to extract features directly from raw s1 and candidate dictionaries.
    """
    raw_n1 = str(s1_record.get('business_name', '')) if pd.notna(s1_record.get('business_name')) else ""
    raw_n2 = str(candidate_record.get('business_name', '')) if pd.notna(candidate_record.get('business_name')) else ""
    raw_a1 = str(s1_record.get('business_address', '')) if pd.notna(s1_record.get('business_address')) else ""
    raw_a2 = str(candidate_record.get('business_address', '')) if pd.notna(candidate_record.get('business_address')) else ""

    c1 = str(s1_record.get('country', ''))
    c2 = str(candidate_record.get('country', ''))
    target_id = str(candidate_record.get('entity_id', ''))

    norm_n1 = normalize_name(raw_n1)
    norm_n2 = normalize_name(raw_n2)
    norm_a1 = normalize_address(raw_a1)
    norm_a2 = normalize_address(raw_a2)

    toks_n1 = set(norm_n1.split())
    toks_n2 = set(norm_n2.split())
    toks_a1 = set(norm_a1.split())
    toks_a2 = set(norm_a2.split())

    return extract_pair_features_precomputed(
        norm_n1, toks_n1, norm_a1, toks_a1, c1,
        norm_n2, toks_n2, norm_a2, toks_a2, c2,
        target_id,
        skip_slow_sim=True
    )
