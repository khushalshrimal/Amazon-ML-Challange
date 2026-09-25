"""
Candidate generation (blocking) module implementing Strategy 3 Multi-Key Union.
Combines:
1. Name 2-character prefix + country index
2. Address 2-character prefix + country index
3. Rare name token + country index
4. Informative/rare address token + country index
5. Name 3-gram + country index
"""

from typing import Dict, List, Set
from collections import defaultdict, Counter

try:
    from .normalize import normalize_name, normalize_address
except ImportError:
    from normalize import normalize_name, normalize_address

def extract_3grams(text: str) -> list:
    if len(text) < 3:
        return [text] if text else []
    return [text[i:i+3] for i in range(len(text) - 2)]

class EntityBlocker:
    """
    Multi-Key Union Blocker (Strategy 3) achieving ~98.65% candidate recall.
    """
    def __init__(self, max_name_token_ratio: float = 0.03, max_addr_token_ratio: float = 0.01, max_ngram_ratio: float = 0.05):
        self.max_name_token_ratio = max_name_token_ratio
        self.max_addr_token_ratio = max_addr_token_ratio
        self.max_ngram_ratio = max_ngram_ratio

        self.name_pref2_idx = defaultdict(set)
        self.addr_pref2_idx = defaultdict(set)
        self.name_rare_tok_idx = defaultdict(set)
        self.addr_rare_tok_idx = defaultdict(set)
        self.name_ngram_idx = defaultdict(set)

        self.stop_name_tokens = set()
        self.stop_addr_tokens = set()
        self.stop_ngrams = set()

    def build_index(self, target_records: list):
        """
        Build inverted indexes over target entity records (S2 and S3).
        """
        self.name_pref2_idx.clear()
        self.addr_pref2_idx.clear()
        self.name_rare_tok_idx.clear()
        self.addr_rare_tok_idx.clear()
        self.name_ngram_idx.clear()

        name_token_freq = Counter()
        addr_token_freq = Counter()
        name_ngram_freq = Counter()

        normalized_targets = []
        for rec in target_records:
            mid = rec['entity_id']
            c = rec.get('country', 'UNKNOWN')
            if not c or c != c:
                c = 'UNKNOWN'
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

        max_n_freq = len(target_records) * self.max_name_token_ratio
        max_a_freq = len(target_records) * self.max_addr_token_ratio
        max_ng_freq = len(target_records) * self.max_ngram_ratio

        self.stop_name_tokens = {t for t, cnt in name_token_freq.items() if cnt > max_n_freq}
        self.stop_addr_tokens = {t for t, cnt in addr_token_freq.items() if cnt > max_a_freq}
        self.stop_ngrams = {ng for ng, cnt in name_ngram_freq.items() if cnt > max_ng_freq}

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

    def get_candidates(self, s1_record: dict) -> Set[str]:
        """
        Retrieve candidate matching target entity IDs for a given S1 record using Strategy 3 Multi-Key Union.
        """
        c = s1_record.get('country', 'UNKNOWN')
        if not c or c != c:
            c = 'UNKNOWN'

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
