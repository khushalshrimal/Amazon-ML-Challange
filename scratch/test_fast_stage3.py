import os
import sys
import time
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from normalize import normalize_name, normalize_address

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

# Load model
model_path = base_dir / "models" / "lightgbm_model.joblib"
model = joblib.load(model_path)
print("Model loaded successfully.")

# Test feature equivalence on a dummy pair
s1 = {
    'norm_n': 'amazon standard store',
    'norm_a': '123 main street seattle wa',
    'c': 'us',
    'toks_n': {'amazon', 'standard', 'store'},
    'toks_a': {'123', 'main', 'street', 'seattle', 'wa'},
    'ngrams_n': extract_ngrams('amazon standard store', 3),
    'ngrams_a': extract_ngrams('123 main street seattle wa', 3),
    'digits_a': {'123'},
    'len_n': len('amazon standard store'),
    'len_a': len('123 main street seattle wa'),
    'len_toks_n': 3,
    'len_toks_a': 5,
    'len_ng_n': len(extract_ngrams('amazon standard store', 3)),
    'len_ng_a': len(extract_ngrams('123 main street seattle wa', 3)),
    'len_dig_a': 1,
}

t_parsed = (
    'us', # country
    'amazon standard store inc', # norm_n
    '123 main st seattle wa 98101', # norm_a
    {'amazon', 'standard', 'store', 'inc'},
    {'123', 'main', 'st', 'seattle', 'wa', '98101'},
    extract_ngrams('amazon standard store inc', 3),
    extract_ngrams('123 main st seattle wa 98101', 3),
    {'123', '98101'},
    len('amazon standard store inc'),
    len('123 main st seattle wa 98101'),
    4, 6,
    len(extract_ngrams('amazon standard store inc', 3)),
    len(extract_ngrams('123 main st seattle wa 98101', 3)),
    2,
    1.0
)

# Reference feature extraction
c2, norm_n2, norm_a2, toks_n2, toks_a2, ngrams_n2, ngrams_a2, digits_a2 = t_parsed[0], t_parsed[1], t_parsed[2], t_parsed[3], t_parsed[4], t_parsed[5], t_parsed[6], t_parsed[7]
len_n2, len_a2, len_tn2, len_ta2, len_ng2_n, len_ng2_a, len_d2, is_s2 = t_parsed[8], t_parsed[9], t_parsed[10], t_parsed[11], t_parsed[12], t_parsed[13], t_parsed[14], t_parsed[15]

n_exact = 1.0 if s1['norm_n'] == norm_n2 and s1['norm_n'] != "" else 0.0
len_tn1 = s1['len_toks_n']
if len_tn1 > 0 and len_tn2 > 0:
    n_overlap = len(s1['toks_n'] & toks_n2)
    n_jaccard = n_overlap / (len_tn1 + len_tn2 - n_overlap)
else: n_overlap, n_jaccard = 0, 0.0

len_ng1_n = s1['len_ng_n']
if len_ng1_n > 0 and len_ng2_n > 0:
    ng_n_overlap = len(s1['ngrams_n'] & ngrams_n2)
    n_char_jaccard = ng_n_overlap / (len_ng1_n + len_ng2_n - ng_n_overlap)
else: n_char_jaccard = 0.0

len_ta1 = s1['len_toks_a']
if len_ta1 > 0 and len_ta2 > 0:
    a_overlap = len(s1['toks_a'] & toks_a2)
    a_jaccard = a_overlap / (len_ta1 + len_ta2 - a_overlap)
else: a_overlap, a_jaccard = 0, 0.0

len_ng1_a = s1['len_ng_a']
if len_ng1_a > 0 and len_ng2_a > 0:
    ng_a_overlap = len(s1['ngrams_a'] & ngrams_a2)
    a_char_jaccard = ng_a_overlap / (len_ng1_a + len_ng2_a - ng_a_overlap)
else: a_char_jaccard = 0.0

len_d1 = s1['len_dig_a']
if len_d1 > 0 and len_d2 > 0:
    d_overlap = len(s1['digits_a'] & digits_a2)
    d_jaccard = d_overlap / (len_d1 + len_d2 - d_overlap)
else: d_jaccard = 0.5

c_match = 1.0 if s1['c'] == c2 and s1['c'] != "" else 0.0
len_diff_n = float(abs(s1['len_n'] - len_n2))
len_diff_a = float(abs(s1['len_a'] - len_a2))
missing_addr = 1.0 if s1['len_a'] == 0 or len_a2 == 0 else 0.0

feats = np.array([[n_exact, n_jaccard, n_char_jaccard, a_jaccard, a_char_jaccard, d_jaccard, c_match, len_diff_n, len_diff_a, n_overlap, a_overlap, missing_addr, is_s2]], dtype=np.float32)
prob = model.predict_proba(feats)[0, 1]
print("Computed features:", feats)
print(f"Predicted probability: {prob:.4f} (Match >= 0.98: {prob >= 0.98})")
