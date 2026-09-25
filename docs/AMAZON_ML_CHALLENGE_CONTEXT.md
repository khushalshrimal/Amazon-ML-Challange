# Amazon ML Challenge 2026 — Verified Challenge Context & Project Plan

## PART 1: OFFICIAL RULES & VERIFIED CHALLENGE SPECIFICATIONS

### 1. Official Problem Statement
* **Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge
* **Goal:** Identify which records from Source 2 (`S2-`) and Source 3 (`S3-`) refer to the same real-world business entity as each reference record in Source 1 (`S1-`).
* **Source Cardinality:**
  * Source 1 (`S1-`): Deduplicated reference source.
  * Source 2 (`S2-`) & Source 3 (`S3-`): Potential matching records.
  * Each Source 1 entity may match **zero** (singleton), **one**, or **multiple** records across Source 2 and/or Source 3.

### 2. Verified Dataset Structure & File Statistics
All data files are located in `6ab10eb3b23ba_student_resource/student_resource/dataset/`.

| File | Path | File Size | Exact Record Count | Columns | Country Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | `dataset/train/train_source1.tsv` | 200.34 MB | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` | US: 1,323,633 (60.0%), India: 883,188 (40.0%) |
| `train_source2.tsv` | `dataset/train/train_source2.tsv` | 466.63 MB | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` | US: 3,016,817 (59.9%), India: 2,017,799 (40.1%) |
| `train_source3.tsv` | `dataset/train/train_source3.tsv` | 480.37 MB | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` | US: 3,170,056 (60.0%), India: 2,115,547 (40.0%) |
| `train_ground_truth.tsv` | `dataset/train/train_ground_truth.tsv` | 121.13 MB | 2,206,821 | `source1_entity_id`, `matched_entity_ids` | Matches: 2,083,574 (94.42%), Singletons: 123,247 (5.58%) |
| `test_source1.tsv` | `dataset/test/test_source1.tsv` | 166.91 MB | 1,732,544 | `entity_id`, `business_name`, `business_address`, `country` | India: 809,986 (46.8%), US: 663,106 (38.3%), France: 259,452 (15.0%) |
| `test_source2.tsv` | `dataset/test/test_source2.tsv` | 485.86 MB | 4,887,273 | `entity_id`, `business_name`, `business_address`, `country` | India: 2,312,565 (47.3%), US: 1,871,330 (38.3%), France: 703,378 (14.4%) |
| `test_source3.tsv` | `dataset/test/test_source3.tsv` | 482.56 MB | 5,082,316 | `entity_id`, `business_name`, `business_address`, `country` | India: 2,405,000 (47.3%), US: 1,945,701 (38.3%), France: 731,615 (14.4%) |

### 3. Open-Set Country Rule
* **Training Data:** US and India only.
* **Test Data:** US, India, and **France** (259,452 S1 test entities, 703,378 S2 test entities, 731,615 S3 test entities).
* **Mandatory Requirements:**
  1. Never hard-code pipelines or encodings to only `{US, India}`.
  2. Never filter out France records.
  3. Treat `country` as an open set of string labels.
  4. Every test Source 1 entity (including all France entities) must appear in the final submission output.

### 4. Official Submission Outputs & Validation Rules
* **Format:** Tab-separated (`.tsv`) with explicit `\t` delimiter.
* **Outputs:**
  1. `output/matching_results.tsv` (Scored on Leaderboard):
     * Columns: `source1_entity_id`, `matched_entity_ids`
     * Exactly 1,732,544 rows matching every `S1-` entity in `test_source1.tsv`.
     * `matched_entity_ids` is comma-separated without quotes (or empty for singletons).
  2. `output/candidate_pairs.tsv` (Included in Submission Zip):
     * Columns: `source1_entity_id`, `candidate_entity_ids`
     * Contains the exact candidate set fed into model inference.
     * All matched IDs in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`.
* **Validation:** Verified locally using `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.

### 5. Official Evaluation Metric
* **Metric:** Macro-averaged $F_{0.5}$ score.
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Characteristics:** Precision is weighted $2\times$ over recall. False merges (matching different businesses) are penalized heavily. Singletons with empty match lists score 1.0 if correctly predicted empty, 0.0 if false match predicted.

### 6. Strict Challenge Constraints & Model Restrictions
* **External Data:** Strictly prohibited (no Google Maps, geocoding APIs, external DBs, web search lookups).
* **Model Restrictions:** Must use models with **MIT or Apache 2.0 license** and up to **8 Billion parameters**.
* **Submission Limits:** 5 submissions per day, maximum 15 over 3 days.

---

## PART 2: OUR PLANNED TECHNICAL APPROACH

### Phase 1: Environment & Directory Setup
* Organize repository into modular pipeline structure under `code/business_entity_resolution/`.
* Set up reproducible data loading scripts reading TSVs with `sep="\t"`.

### Phase 2: EDA & Noise Profiling
* Analyze string noise patterns across US, India, and France records (abbreviations, legal suffixes, typos, PIN code formats, landmark addresses).

### Phase 3: Validation Framework
* Create representative local validation split from `train_source1.tsv` and `train_ground_truth.tsv`.
* Implement macro $F_{0.5}$, precision, recall, and singleton accuracy metrics matching official evaluation.

### Phase 4: High-Recall Blocking / Candidate Generation
* Develop multi-tier blocking strategies (e.g. country blocking, token prefix index, min-hash / LSH, character n-gram index) to maximize blocking recall while reducing $1.7\text{M} \times 9.9\text{M}$ candidate space to manageable pairs.

### Phase 5: Feature Engineering & Distance Metrics
* Extract name similarity (Jaccard, Levenshtein, Jaro-Winkler, TF-IDF cosine, length ratios).
* Extract address similarity (token overlap, PIN code equality, character n-gram cosine).

### Phase 6: Matching Model & Threshold Tuning
* Train efficient, license-compliant GBDT models (LightGBM/XGBoost/CatBoost) or calibrated classifiers.
* Optimize prediction threshold specifically for macro $F_{0.5}$ on validation set.

### Phase 7: Inference, Validation & Submission Packaging
* Generate `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
* Pass all checks with `utils/validate_submission.py`.
* Package final zip archive with reproducible code and filled methodology documentation.
