# Submission 2 Experiment Log — Amazon ML Challenge 2026

| Exp ID | Hypothesis / Description | Blocker Config | Model | Features | Threshold | Candidate Recall | Macro F0.5 | Precision | Recall | Candidates/S1 | Runtime | RAM | Result / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_000** | Baseline: Submission 1 pipeline | PostLimit=100, Pref=3 | LightGBM Baseline | 13 Base Features | 0.980 | 98.83% | 0.9410 | 0.9552 | 0.9205 | 348.5 | 119.1s | 3.89GB | **BASELINE** |
| **EXP_001** | Multi-Key Union Blocker Optimization | PostLimit=100, Pref=3, StopWords | Blocker Benchmark | N/A | N/A | **98.83%** | N/A | N/A | N/A | 348.5 | 2.08s | 0.89GB | **ACCEPTED** — High recall, eliminates explosion |
| **EXP_002** | LightGBM v2 + 32 Features + Hard Negatives | PostLimit=100, Pref=3 | LightGBM v2 (Tuned) | 32 Features v2 | 0.997 | 98.83% | 0.9596 | 0.9687 | 0.9461 | 348.5 | 14.2s | 1.48GB | **ACCEPTED** — +0.0186 F0.5 boost |
| **EXP_003** | XGBoost v2 + 32 Features + Hard Negatives | PostLimit=100, Pref=3 | XGBoost v2 (Tuned) | 32 Features v2 | 0.992 | 98.83% | **0.9613** | 0.9710 | 0.9460 | 348.5 | 9.5s | 1.48GB | **ACCEPTED** — +0.0203 F0.5 boost |
| **EXP_004** | Ensemble (0.5 LightGBM + 0.5 XGBoost) | PostLimit=100, Pref=3 | Ensemble Blend | 32 Features v2 | 0.995 | 98.83% | **0.9613** | **0.9718** | 0.9433 | 348.5 | 16.5s | 1.56GB | **WINNER (FROZEN)** — Highest Precision & F0.5 |

---
