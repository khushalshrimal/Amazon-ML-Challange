# Business Entity Resolution — Source Pipeline

## Overview
This directory contains the self-contained modular source code for the Amazon ML Challenge 2026 Business Entity Resolution submission.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── config.py             # Hyperparameters, file paths, open-set country settings
│   ├── data_loader.py        # Memory-aware TSV data loaders (sep='\t')
│   ├── normalize.py         # Business name & address text standardization
│   ├── blocking.py          # Candidate generation & search space reduction
│   ├── features.py          # String/address similarity feature extraction
│   ├── model.py             # Match classification model (GBDT)
│   ├── evaluate.py          # Local macro F0.5 evaluation metric calculation
│   ├── predict.py           # Inference & output TSV generation
│   ├── pipeline.py          # End-to-end pipeline execution entrypoint
│   └── validate_submission.py # Official submission validator copy
├── README.md                 # End-to-end reproduction instructions
└── requirements.txt         # Pinned python environment dependencies
```

## Reproduction Instructions
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end pipeline:
   ```bash
   python -m src.pipeline
   ```
3. Validate output format using official validator:
   ```bash
   python src/validate_submission.py --matching "../../output/matching_results.tsv" --candidate "../../output/candidate_pairs.tsv" --test-dir "../../dataset/test"
   ```
