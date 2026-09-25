"""
Candidate generation (blocking) module implementing high-recall multi-pass blocking.
"""

from typing import Dict, List, Set
from collections import defaultdict, Counter

try:
    from .normalize import normalize_name, normalize_address
except ImportError:
    from normalize import normalize_name, normalize_address

class EntityBlocker:
    """
    Multi-pass blocking engine for entity resolution search space reduction.
    Combines:
    1. 2-word name prefix + country index
    2. 2-word address prefix + country index
    3. Rare name token + country inverted index
    """
    def __init__(self, max_token_freq_ratio: float = 0.03):
        self.max_token_freq_ratio = max_token_freq_ratio
        self.country_index = defaultdict(set)
        self.name_pref2_index = defaultdict(set)
        self.addr_pref2_index = defaultdict(set)
        self.token_index = defaultdict(set)
        self.stop_tokens = set()

    def build_index(self, target_records: list):
        """
        Build inverted indexes over target entity records (S2 and S3).
        target_records: list of dicts or records with keys ['entity_id', 'business_name', 'business_address', 'country']
        """
        self.country_index.clear()
        self.name_pref2_index.clear()
        self.addr_pref2_index.clear()
        self.token_index.clear()
        self.stop_tokens.clear()

        token_freq = Counter()
        normalized_targets = []

        for rec in target_records:
            mid = rec['entity_id']
            country = rec.get('country', 'UNKNOWN')
            if not country or country != country:
                country = 'UNKNOWN'

            raw_name = str(rec.get('business_name', '')) if rec.get('business_name') is not None else ""
            raw_addr = str(rec.get('business_address', '')) if rec.get('business_address') is not None else ""

            norm_n = normalize_name(raw_name)
            norm_a = normalize_address(raw_addr)

            n_toks = norm_n.split()
            token_freq.update(n_toks)
            normalized_targets.append((mid, country, norm_n, norm_a, n_toks))

        max_freq = len(target_records) * self.max_token_freq_ratio
        self.stop_tokens = {tok for tok, cnt in token_freq.items() if cnt > max_freq}

        for mid, c, norm_n, norm_a, n_toks in normalized_targets:
            self.country_index[c].add(mid)

            if len(n_toks) >= 2:
                self.name_pref2_index[(" ".join(n_toks[:2]), c)].add(mid)
            elif n_toks:
                self.name_pref2_index[(n_toks[0], c)].add(mid)

            a_toks = norm_a.split()
            if len(a_toks) >= 2:
                self.addr_pref2_index[(" ".join(a_toks[:2]), c)].add(mid)
            elif a_toks:
                self.addr_pref2_index[(a_toks[0], c)].add(mid)

            for tok in [t for t in n_toks if t not in self.stop_tokens and len(t) > 2][:2]:
                self.token_index[(tok, c)].add(mid)

    def get_candidates(self, s1_record: dict) -> Set[str]:
        """
        Retrieve candidate matching target entity IDs for a given S1 record.
        """
        c = s1_record.get('country', 'UNKNOWN')
        if not c or c != c:
            c = 'UNKNOWN'

        raw_name = str(s1_record.get('business_name', '')) if s1_record.get('business_name') is not None else ""
        raw_addr = str(s1_record.get('business_address', '')) if s1_record.get('business_address') is not None else ""

        norm_n = normalize_name(raw_name)
        norm_a = normalize_address(raw_addr)

        cands = set()

        n_toks = norm_n.split()
        if len(n_toks) >= 2:
            cands.update(self.name_pref2_index.get((" ".join(n_toks[:2]), c), set()))
        elif n_toks:
            cands.update(self.name_pref2_index.get((n_toks[0], c), set()))

        a_toks = norm_a.split()
        if len(a_toks) >= 2:
            cands.update(self.addr_pref2_index.get((" ".join(a_toks[:2]), c), set()))
        elif a_toks:
            cands.update(self.addr_pref2_index.get((a_toks[0], c), set()))

        for tok in [t for t in n_toks if t not in self.stop_tokens and len(t) > 2][:2]:
            cands.update(self.token_index.get((tok, c), set()))

        return cands
