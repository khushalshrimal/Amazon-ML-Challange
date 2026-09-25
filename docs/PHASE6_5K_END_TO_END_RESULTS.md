# Phase 6 — Final 5K End-to-End Validation Results

## 1. Overview & Dataset Statistics
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
| Candidate Generation (blocking) | 38.24 s |
| Feature Extraction (feature_extraction) | 145.71 s |
| Model Training (model_training) | 4.45 s |
| Threshold Evaluation (threshold_evaluation) | 19.10 s |
| **Total End-to-End Runtime** | **243.78 s** |

---

## 3. Decision Threshold Grid Search
| Threshold ($T$) | Macro $F_{0.5}$ | Precision | Recall | Total Predicted Matches | Zero-Match S1 Entities |
|---|---|---|---|---|---|
| 0.80 | **0.8177** | 0.8048 | 0.9500 | 7,584 | 44 |
| 0.85 | **0.8379** | 0.8277 | 0.9493 | 7,111 | 49 |
| 0.90 | **0.8624** | 0.8557 | 0.9487 | 6,643 | 54 |
| 0.92 | **0.8743** | 0.8695 | 0.9476 | 6,422 | 58 |
| 0.94 | **0.8842** | 0.8813 | 0.9455 | 6,204 | 60 |
| 0.95 | **0.8907** | 0.8889 | 0.9445 | 6,079 | 62 |
| 0.96 | **0.8976** | 0.8984 | 0.9390 | 5,905 | 62 |
| 0.97 | **0.9153** | 0.9197 | 0.9372 | 5,589 | 67 |
| 0.98 | **0.9301** | 0.9394 | 0.9308 | 5,318 | 72 |

---

## 4. Final Validation Summary & Achievements
* **Candidate Recall Milestone:** Strategy 3 Multi-Key Union achieved an impressive **98.64% Candidate Recall** on 5,000 validation entities.
* **End-to-End Accuracy:** At threshold **$T^* = 0.98$**, the model achieved **Macro $F_{0.5} = 0.9301$** (Precision = 0.9394, Recall = 0.9308).
* **Execution Efficiency:** Entire end-to-end 5k pipeline executed in **243.78 seconds** without any dynamic programming bottlenecks.
