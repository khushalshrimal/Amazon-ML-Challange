# SUBMISSION 2 AUTONOMOUS RESUME CONTRACT

**Status**: `PAUSED_SAFE`  
**Timestamp**: 2026-09-26T17:22:30+05:30  
**Project Directory**: `C:\Users\khush\OneDrive\Desktop\amzon ml challenge`  

---

## 1. CURRENT STATE SUMMARY

- **Current Stage**: Stage A/B Two-Stage Retrieval Benchmark & Full-Test Inference Checkpoint
- **Processed S1**: 5,000 / 1,732,544 (0.29%)
- **Candidates Processed**: 1,798,101 pairs (359.6 candidates/S1)
- **Matched S1 Count**: 4,541 / 5,000 (90.82% match rate)
- **Total Matches Found**: 19,113 matches
- **Models Loaded**: `models/lightgbm_model_v2.joblib` & `models/xgboost_model_v2.joblib`
- **Threshold**: 0.995 (Ensemble Blend)
- **Validation Score**: **Macro F0.5 = 0.9613** (Macro Precision = 0.9718, Macro Recall = 0.9433)

---

## 2. REUSABLE ARTIFACTS MANIFEST

The following artifacts exist on disk and MUST be reused upon resuming:

1. **Submission 1 (FROZEN)**: `submission/Submission_1/matching_results.tsv` (Leaderboard score: 0.522 - IMMUTABLE)
2. **Leakage-Free Validation Split**: `scratch/val_split/val_s1.tsv`, `val_targets.tsv`, `val_gt.tsv`
3. **Trained Models**:
   - `models/lightgbm_model_v2.joblib`
   - `models/xgboost_model_v2.joblib`
4. **Validation Features Cache**: `scratch/val_features_cache_v2.joblib` (462.5 MB)
5. **Full Test Inference Checkpoint**: `scratch/inference_checkpoint.json` (`{"processed_s1": 5000}`)
6. **Partial Output Files**:
   - `output/matching_results.tsv` (5,001 lines, 316.3 KB)
   - `output/candidate_pairs.tsv` (5,001 lines, 23.2 MB)
7. **Submission 2 Config**: `submission/Submission_2/final_config.json`

---

## 3. RESUME CONTRACT INSTRUCTIONS

When the user returns and types `CONTINUE`:

1. Read `scratch/autonomous_state.json` and `scratch/RESUME.md`.
2. Inspect `scratch/artifact_manifest.json` and verify all disk paths exist.
3. Execute the 100-S1 Fast Benchmark Gate (runtime <= 30s per 100 S1).
4. Resume full test set inference from `processed_s1 = 5000` using multi-core Top-K two-stage parallel retrieval.
5. Do NOT restart from scratch or discard completed output lines.
6. Do NOT modify Submission 1 files.
7. Run `py -3 utils/validate_submission.py` to validate outputs.
8. Archive Submission 2 to `submission/Submission_2/` and present final summary report.
