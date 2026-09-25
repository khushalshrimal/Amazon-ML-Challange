# Baseline Feature Analysis Report — Amazon ML Challenge 2026

## 1. Executive Summary
This report analyzes baseline feature discriminative power by comparing similarity features extracted over 2,000 true matched entity pairs ($y=1$) against 2,000 random candidate non-matches ($y=0$).

---

## 2. Feature Mean Comparison Table

| Feature Name | Description | True Matches ($y=1$) | Non-Matches ($y=0$) | Discriminative Power |
| :--- | :--- | :---: | :---: | :--- |
| `name_jaccard` | Business Name Token Overlap | **0.6496** | **0.0146** | **Extremely Strong** ($44.5\times$ separation) |
| `name_char_ratio` | Character Sequence Ratio (SequenceMatcher) | **0.7929** | **0.2506** | **Strong** ($3.16\times$ separation) |
| `addr_jaccard` | Business Address Token Overlap | **0.6122** | **0.0095** | **Extremely Strong** ($64.4\times$ separation) |
| `addr_char_ratio` | Character Sequence Ratio for Address | **0.7546** | **0.2311** | **Strong** ($3.27\times$ separation) |
| `name_len_diff` | Absolute Name Length Difference | **3.85 chars** | **9.33 chars** | **Moderate** ($2.42\times$ tighter) |
| `addr_len_diff` | Absolute Address Length Difference | **10.79 chars** | **26.83 chars** | **Moderate** ($2.49\times$ tighter) |
| `country_match` | Exact Country Label Match | **1.0000** | **1.0000** (within-country) | **Mandatory Filter** |

---

## 3. Key Observations for Candidate Feature Engineering
1. **Token Overlap vs Sequence Similarity:**
   * Token Jaccard similarity (`name_jaccard`, `addr_jaccard`) provides the cleanest linear separation between true matches and non-matches (0.65 vs 0.01).
   * Character ratio (`name_char_ratio`) excels at catching single-character typos and spelling variants where token Jaccard fails.
2. **Address Similarity Discriminative Power:**
   * Address similarity features are nearly as strong as name similarity features (0.61 vs 0.01). Combining name and address similarities produces near-orthogonal signals for model classification.
3. **Length Differences:**
   * True matches have an average name length difference of only 3.85 characters compared to 9.33 characters for random candidate pairs.
