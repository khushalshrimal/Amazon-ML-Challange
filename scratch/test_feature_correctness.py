import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"))

from features_v2 import fill_features_v2, precompute_s1_v2, FEATURE_NAMES_V2, OnDemandTargetStoreV2

def test_feature_equivalence():
    print("Testing Feature Equivalence...")
    target_store = OnDemandTargetStoreV2()
    target_store.add_target_raw("S2-100", "US", "acme corporation llc", "123 main street new york ny 10001")
    target_store.add_target_raw("S3-200", "US", "beta industries", "456 park avenue chicago il 60601")

    s1_rec = {
        "entity_id": "S1-1",
        "country": "US",
        "business_name": "acme corp",
        "business_address": "123 main st ny"
    }

    s1_prep = precompute_s1_v2(s1_rec["entity_id"], s1_rec["country"], s1_rec["business_name"], s1_rec["business_address"])

    X1 = np.zeros((2, len(FEATURE_NAMES_V2)), dtype=np.float32)
    fill_features_v2(X1, 0, s1_prep, target_store, "S2-100")
    fill_features_v2(X1, 1, s1_prep, target_store, "S3-200")

    print(f"Features computed successfully shape: {X1.shape}")
    print("Row 0 features:", X1[0])
    print("Row 1 features (far candidate):", X1[1])
    assert X1[1][3] == 0.0 # rf_ratio
    assert X1[1][16] == 0.0 # addr_rf_ratio
    print("ALL FEATURE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_feature_equivalence()
