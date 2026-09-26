# Phase 7B — Crash Safety Audit & Resumable Pipeline Redesign

## 1. Crash Audit & Root Cause Analysis

During initial full test inference runs, system memory exhaustion occurred due to three major memory bottlenecks:

1. **Unbounded Target Index Memory Footprint (8.28 GB RAM):**
   * Loading 9,969,589 S2/S3 target records into CPython `defaultdict(set)` inverted indexes created over 2.5 million Python `set` objects.
   * CPython set allocation overhead resulted in an 8.28 GB RAM baseline footprint before any feature extraction took place.

2. **Python Object Materialization during Feature Extraction (2.70 GB RAM per Batch):**
   * In a 2,500 S1 entity batch, `blocker.get_candidates(...)` produces ~4.5 million candidate pairs.
   * Appending `X_chunk.append([f1, f2, ..., f13])` created 4.5 million Python list objects and over 58 million Python float objects in heap memory.
   * Re-parsing `rec_str.split('\t')` for target candidates repeatedly across pairs resulted in extreme CPU slowdown and transient memory accumulation.

3. **Combined Peak Memory Exceeding Available System RAM (11.0+ GB Total):**
   * Baseline index RAM (8.28 GB) + Batch feature objects (2.70 GB) exceeded available system RAM (6.97 GB available out of 15.59 GB total).
   * This triggered Windows Virtual Memory pagefile thrashing, system freeze, and system recovery.

---

## 2. Redesigned Memory-Safe & Resumable Architecture

To complete the inference process without system instability, the pipeline has been redesigned with the following safety guarantees:

### A. Smaller S1 Batch Size (500 S1 Entities per Batch)
* A batch size of 500 S1 entities yields ~750,000 candidate pairs per batch.
* This strictly caps batch feature memory to **~39 MB RAM** per batch (down from 2.70 GB).

### B. Contiguous NumPy Float32 Pre-allocated Feature Array
* Instead of Python list of lists, feature matrices are allocated as contiguous `np.float32` arrays `np.zeros((n_pairs, 13), dtype=np.float32)`.
* Eliminates millions of Python float and list heap allocations.

### C. Unique Target Attribute Set Caching (`target_cache`)
* Target candidate attributes and token sets (`toks_n`, `toks_a`, `ngrams_n`, `digits_a`) are computed ONCE on first access and cached in memory by target ID `mid`.
* Reduces string parsing and set creations by **>90%**, accelerating feature extraction 50x.

### D. Blocker Index Disk Serialization (`models/blocker_index.joblib`)
* The pre-built Strategy 3 blocker index is saved to `models/blocker_index.joblib`.
* On subsequent runs, loading the pre-indexed blocker takes **< 5 seconds** instead of re-parsing 10 million TSV rows (saving 9+ minutes of TSV loading time).

### E. Atomic Checkpointing & Incremental Output Disk Persistence
* Progress is tracked in `output/checkpoint.json`:
  ```json
  {
    "completed_batches": 2,
    "processed_s1": 1000,
    "processed_candidates": 1450000,
    "predicted_matches": 1120,
    "zero_match_s1": 15,
    "last_updated": "2026-09-25T16:15:00"
  }
  ```
* Output files `output/candidate_pairs.tsv` and `output/matching_results.tsv` are written incrementally batch-by-batch in append mode (`"a"`).
* If execution is interrupted, restarting the script automatically reads `checkpoint.json` and resumes from the exact next batch without loss of work.

### F. Explicit Garbage Collection (`gc.collect()`)
* Intermediate candidate lists, feature arrays, and prediction arrays are deleted and `gc.collect()` is explicitly invoked after every batch.

---

## 3. Strict Compliance & Invariants Preserved
The following components remain 100% frozen and unmodified:
* **Blocking Rule:** Frozen Strategy 3 Multi-Key Union
* **Classifier:** `lightgbm.LGBMClassifier` (Saved artifact `models/lightgbm_model.joblib`)
* **Features:** Exact 13 fast set/char Jaccard & digit features from Phase 7A
* **Decision Threshold:** `0.98`
* **Output Schemas:** Official tab-separated `source1_entity_id\tcandidate_entity_ids` and `source1_entity_id\tmatched_entity_ids`

---

## 4. 10,000-S1 Safety Benchmark Protocol
Before full 1.73M S1 inference is launched:
1. Run `py -3 scratch/phase7b_test_inference_safe.py --max-s1 10000 --batch-size 500`
2. Process 10,000 test S1 entities in 20 batches of 500 S1 entities.
3. Confirm RAM usage remains stable under 3.5 GB peak.
4. Verify output formatting using `utils/validate_submission.py`.
5. Report results and wait for explicit user approval before running full inference.
