# Phase 4 — 5,000 Entity Validation Run Results

## 1. Overview & Dataset Statistics
* **Validation S1 Sample Size:** 5,000 Ground Truth S1 entities (3,500 Train / 1,500 Held-out Validation).
* **Target Pool Size:** 316,757 S2/S3 unique records.
* **Ground-Truth True Pairs:** 17,250 true match pairs in 5,000 S1 sample.
* **Candidate Pairs Generated:** 4,921,088 candidates (Avg: 984.22 candidates/S1).
* **Candidate Recall:** **91.22%** (15,736 / 17,250 true pairs).
* **Training Pair Sample:** 12,053 positive pairs, 41,013 sampled hard negatives (Total: 53,066).

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

## 3. Stage Runtimes
| Stage | Runtime (seconds) |
|---|---|
| Candidate Generation (blocking) | 30.12 s |
| Feature Extraction (feature_extraction) | 89.02 s |
| Model Training (model_training) | 3.59 s |
| Threshold Evaluation (threshold_evaluation) | 9.59 s |
| **Total Validation Pipeline Runtime** | **132.32 s** |

---

## 4. Threshold Evaluation Results
| Threshold ($T$) | Macro $F_{0.5}$ | Precision | Recall |
|---|---|---|---|
| 0.70 | **0.8447** | 0.8325 | 0.9589 |
| 0.75 | **0.8618** | 0.8518 | 0.9567 |
| 0.80 | **0.8816** | 0.8742 | 0.9577 |
| 0.85 | **0.9060** | 0.9023 | 0.9578 |
| 0.90 | **0.9277** | 0.9276 | 0.9571 |
| 0.95 | **0.9472** | 0.9531 | 0.9489 |

---

## 5. Threshold Analysis & Key Findings
* **Threshold Stability Check:** Decision threshold **$T^* = 0.95$** achieved optimal **Macro $F_{0.5} = 0.9472$** (Precision = 0.9531, Recall = 0.9489).
* **Consistency with Sanity Test:** Threshold $0.90$ remains highly consistent and effective when scaled from 1,000 to 5,000 validation entities.
* **Pipeline Scalability:** Total runtime for 5,000 entities was **132.32 seconds**, proving the fast set-based feature extraction scales predictably.
