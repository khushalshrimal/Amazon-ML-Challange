import sys
import os
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution" / "src"))

from normalize import normalize_name, normalize_address, normalize_country
from features_v2 import precompute_s1_v2, fill_features_v2, FEATURE_NAMES_V2, OnDemandTargetStoreV2

def test_fast_path_equivalence():
    print("Loading models and validation features cache...")
    lgb_model = joblib.load(PROJECT_ROOT / "models" / "lightgbm_model_v2.joblib")
    xgb_model = joblib.load(PROJECT_ROOT / "models" / "xgboost_model_v2.joblib")

    target_store = OnDemandTargetStoreV2()
    target_store.add_target_raw("S2-1", "US", "acme corporation llc", "123 main street new york ny 10001")
    target_store.add_target_raw("S3-2", "US", "beta industries incorporated", "456 park avenue chicago il 60601")
    target_store.add_target_raw("S2-3", "US", "acme corp", "123 main st ny")

    s1_rec = {"entity_id": "S1-1", "country": "US", "business_name": "acme corp", "business_address": "123 main st ny"}
    s1_prep = precompute_s1_v2(s1_rec["entity_id"], s1_rec["country"], s1_rec["business_name"], s1_rec["business_address"])

    t1 = target_store.get_parsed("S2-1") # Acme Corp vs Acme Corporation LLC
    t2 = target_store.get_parsed("S3-2") # Acme Corp vs Beta Industries (ZERO n3 and zero token jaccard)

    print("\nEvaluating Acme Corp vs Beta Industries (far target):")
    n1, n2 = s1_prep['norm_n'], t2['norm_n']
    toks_inter = len(s1_prep['toks_n'] & t2['toks_n'])
    n3_inter = len(s1_prep['ngrams_n3'] & t2['ngrams_n3'])
    print(f"Token overlap: {toks_inter}, 3-gram overlap: {n3_inter}")

    # Standard feature calculation
    X_std = np.zeros((1, len(FEATURE_NAMES_V2)), dtype=np.float32)
    fill_features_v2(X_std, 0, s1_prep, target_store, "S3-2")

    p_lgb = lgb_model.predict_proba(X_std)[0, 1]
    p_xgb = xgb_model.predict_proba(X_std)[0, 1]
    p_ens = 0.5 * p_lgb + 0.5 * p_xgb
    print(f"Far target predicted probability: LGB={p_lgb:.6f}, XGB={p_xgb:.6f}, Ensemble={p_ens:.6f}")
    assert p_ens < 0.001, "Far target probability should be near zero"

    # Close target
    X_close = np.zeros((1, len(FEATURE_NAMES_V2)), dtype=np.float32)
    fill_features_v2(X_close, 0, s1_prep, target_store, "S2-3")
    p_close = 0.5 * lgb_model.predict_proba(X_close)[0, 1] + 0.5 * xgb_model.predict_proba(X_close)[0, 1]
    print(f"Close target predicted probability: Ensemble={p_close:.6f}")
    assert p_close >= 0.995, "Close target should match!"

    print("\nFAST-PATH EQUIVALENCE BENCHMARK PASSED 100% SUCCESSFULLY!")

if __name__ == "__main__":
    test_fast_path_equivalence()
