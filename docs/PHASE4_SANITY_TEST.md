# Phase 4 — 1,000-Entity Sanity Test Results

## 1. Test Overview & Dataset Sizes
* **Sample Size:** 1,000 Ground Truth S1 entities (700 Train / 300 Held-out Validation).
* **Target Pool Size:** 203,382 S2/S3 unique records.
* **Ground-Truth True Pairs:** 3,444 true match pairs in 1,000 S1 sample.
* **Candidate Pairs Generated:** 623,140 candidates (Avg: 623.14 candidates/S1).
* **Candidate Recall:** **92.02%** (3,169 / 3,444 true pairs).
* **Training Pair Sample:** 2,377 positive pairs, 7,989 sampled hard negatives (Total: 10,366).

---

## 2. Features Used
Fast precomputed set intersection & length difference metrics (No `difflib.SequenceMatcher.ratio()`):
1. `name_exact_norm`: Boolean float indicator for exact normalized name match
2. `name_jaccard`: Token Jaccard similarity of normalized business name
3. `name_char_jaccard_3gram`: 3-gram character set Jaccard similarity of normalized name
4. `addr_jaccard`: Token Jaccard similarity of normalized business address
5. `addr_char_jaccard_3gram`: 3-gram character set Jaccard similarity of normalized address
6. `digit_jaccard`: Jaccard similarity of extracted house numbers / zip digits
7. `country_exact_match`: Boolean float for exact country match
8. `name_len_diff`: Absolute character length difference in normalized names
9. `addr_len_diff`: Absolute character length difference in normalized addresses
10. `name_token_overlap`: Integer token overlap count in normalized names
11. `addr_token_overlap`: Integer token overlap count in normalized addresses
12. `missing_address_indicator`: Boolean float indicating empty address
13. `is_s2`: Candidate source indicator (1.0 for S2, 0.0 for S3)

---

## 3. Model Availability Status
* **LightGBM (MIT License):** NOT INSTALLED
* **XGBoost (Apache 2.0 License):** NOT INSTALLED
* **Model Used for Sanity Evaluation:** `HistGradientBoostingClassifier (sklearn / BSD-3-Clause fallback)`

---

## 4. Stage Runtimes
| Stage | Runtime (seconds) |
|---|---|
| Candidate Generation (blocking) | 30.92 s |
| Feature Extraction (feature_extraction) | 19.15 s |
| Model Training (model_training) | 11.27 s |
| Threshold Evaluation (threshold_evaluation) | 2.34 s |
| **Total Sanity Test Runtime** | **63.69 s** |

---

## 5. Threshold Evaluation Results
| Threshold ($T$) | Macro $F_{0.5}$ | Precision | Recall |
|---|---|---|---|
| 0.50 | **0.8253** | 0.8104 | 0.9690 |
| 0.60 | **0.8425** | 0.8306 | 0.9642 |
| 0.70 | **0.8614** | 0.8527 | 0.9610 |
| 0.80 | **0.8818** | 0.8765 | 0.9556 |
| 0.90 | **0.9069** | 0.9058 | 0.9524 |

---

## 6. Summary & Recommendations
* **Pipeline Speed:** The 1,000-entity sanity test completed cleanly in **63.69 seconds** without any runtime bottleneck.
* **Optimal Sanity Threshold:** **$T^* = 0.90$** achieved a **Macro $F_{0.5}$ score of 0.9069** (Precision = 0.9058, Recall = 0.9524).
* **Next Action:** LightGBM (MIT) and XGBoost (Apache 2.0) are NOT currently installed. Run `pip install lightgbm` if strict MIT/Apache 2.0 license compliance is required for final submitted binary.
