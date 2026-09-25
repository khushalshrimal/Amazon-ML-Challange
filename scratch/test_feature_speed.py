import time
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

def fast_jaccard(s1_toks, s2_toks):
    if not s1_toks or not s2_toks:
        return 0.0
    inter = len(s1_toks & s2_toks)
    if inter == 0:
        return 0.0
    return inter / len(s1_toks | s2_toks)

print("Testing Jaccard and SequenceMatcher speed...")
s1_set = {"amazon", "services", "inc"}
s2_set = {"amazon", "retail", "inc", "llc"}
n1 = "amazon services inc"
n2 = "amazon retail inc llc"

t0 = time.time()
for _ in range(100000):
    j = fast_jaccard(s1_set, s2_set)
t1 = time.time()
print(f"100,000 set Jaccard operations took: {t1-t0:.4f}s")

t0 = time.time()
for _ in range(10000):
    r = SequenceMatcher(None, n1, n2).ratio()
t1 = time.time()
print(f"10,000 SequenceMatcher operations took: {t1-t0:.4f}s")
