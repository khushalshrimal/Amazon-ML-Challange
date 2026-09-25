# Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Project Overview
* **Challenge:** Amazon ML Challenge 2026
* **Task:** Business Entity Resolution Challenge — identifying which business records from Source 2 (`S2-`) and Source 3 (`S3-`) refer to the same real-world entity as each reference record in Source 1 (`S1-`).

## 2. Dataset Structure
* **Source 1 (`S1-`):** Reference deduplicated entity records.
* **Source 2 (`S2-`) & Source 3 (`S3-`):** Potential matching business records.
* **Ground Truth (`train_ground_truth.tsv`):** Mapping of `source1_entity_id` to comma-separated `matched_entity_ids` (or empty for singletons).

```
dataset/
├── train/
│   ├── train_source1.tsv       (2,206,821 records — US, India)
│   ├── train_source2.tsv       (5,034,616 records — US, India)
│   ├── train_source3.tsv       (5,285,603 records — US, India)
│   └── train_ground_truth.tsv  (2,206,821 rows — 94.42% matches, 5.58% singletons)
└── test/
    ├── test_source1.tsv        (1,732,544 records — US, India, France)
    ├── test_source2.tsv        (4,887,273 records — US, India, France)
    └── test_source3.tsv        (5,082,316 records — US, India, France)
```

## 3. Important Challenge Constraints & Rules
* **No External Data:** Strictly no external lookup (Google Maps, geocoding APIs, web search, government databases).
* **Open-Set Country Handling:** France appears in the test set (259,452 S1 records) but NOT in training. Never hard-code pipelines to `{US, India}` or filter France. Every S1 test entity must appear in the final submission output.
* **Evaluation Metric:** Macro-averaged $F_{0.5}$ score ($\text{Precision}$ weighted $2\times$ over $\text{Recall}$). False merges (matching incorrect entities) heavily penalize the score. Correct singletons score 1.0.
* **Candidate Set Requirement:** `output/candidate_pairs.tsv` must contain the exact candidate set fed into the matching model for inference. Every matched ID in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
* **Model Constraints:** Final model must be MIT/Apache 2.0 licensed and $\le 8\text{ Billion}$ parameters.

## 4. End-to-End Pipeline Workflow
```
RAW DATA
   ↓
Exploratory Data Analysis (EDA)
   ↓
Text Normalization & Standardisation
   ↓
Candidate Generation / Blocking
   ↓
Candidate Pair Features (Name & Address similarity)
   ↓
Matching Model (GBDT / Calibrated Classifier)
   ↓
Threshold / Decision Logic
   ↓
Local Validation using Macro F0.5
   ↓
Error Analysis & Iterative Refinement
   ↓
Test Candidate & Match Inference
   ↓
Output Files (matching_results.tsv & candidate_pairs.tsv)
   ↓
Official Validator Verification
   ↓
Submission Package ZIP Creation
```

## 5. Official Submission Validation Command
Run the official validator from the project root before submitting:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
