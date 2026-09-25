# Phase 7A — LightGBM Model Validation & Threshold Calibration

## 1. Executive Summary & Compliance
* **Model Class:** `lightgbm.LGBMClassifier` (Version 4.7.0)
* **License Compliance:** MIT License (100% compliant with Amazon ML Challenge rules).
* **Validation S1 Sample Size:** 5,000 Ground Truth S1 entities (3,500 Train / 1,500 Held-out Validation).
* **Target Pool Size:** 316,757 S2/S3 unique records.
* **Ground-Truth True Pairs:** 17,250 true match pairs.
* **Candidate Blocker:** Strategy 3 Multi-Key Union (Frozen).
* **Candidate Pair Count:** 7,288,176 candidate pairs (Avg: 1457.64 candidates/S1).
* **Candidate Recall:** **98.64%** (17,016 / 17,250 true pairs).
* **Training Pair Sample:** 12,053 positive pairs, 42,667 sampled hard negatives (Total: 54,720).

---

## 2. Stage Runtimes
| Stage | Runtime (seconds) |
|---|---|
| Candidate Generation (blocking) | 42.39 s |
| Feature Extraction (feature_extraction) | 194.39 s |
| Model Training (LightGBM fit) | 2.98 s |
| Threshold Evaluation (threshold_evaluation) | 11.56 s |
| **Total End-to-End Runtime** | **294.64 s** |

---

## 3. Decision Threshold Grid Search
| Threshold ($T$) | Macro $F_{0.5}$ | Precision | Recall | Total Predicted Matches | Zero-Match S1 Entities |
|---|---|---|---|---|---|
| 0.80 | 0.8154 | 0.8026 | 0.9482 | 7,501 | 41 |
| 0.85 | 0.8426 | 0.8325 | 0.9491 | 6,940 | 47 |
| 0.90 | 0.8764 | 0.8704 | 0.9498 | 6,352 | 55 |
| 0.92 | 0.8865 | 0.8831 | 0.9467 | 6,143 | 58 |
| 0.94 | 0.8990 | 0.8979 | 0.9447 | 5,921 | 61 |
| 0.95 | 0.9058 | 0.9061 | 0.9438 | 5,800 | 63 |
| 0.96 | 0.9123 | 0.9150 | 0.9397 | 5,656 | 66 |
| 0.97 | 0.9215 | 0.9267 | 0.9375 | 5,479 | 70 |
| 0.98 | **0.9329** | 0.9416 | 0.9335 | 5,301 | 73 |

---

## 4. Key Performance Highlights & Comparison
* **Optimal Decision Threshold:** **$T^* = 0.98$**
* **Macro $F_{0.5}$ Score:** **0.9329** (Precision = 0.9416, Recall = 0.9335)
* **Model Artifact:** Saved to `models/lightgbm_model.joblib`.
* **License Status:** Fully compliant (MIT).

---
