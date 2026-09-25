# Empirical String Noise Analysis Report — Amazon ML Challenge 2026

## 1. Executive Summary
This document analyzes real-world string noise patterns observed across 34,511 true matched entity pairs sampled from `train_ground_truth.tsv`.

---

## 2. Empirical String Match Baselines (34,511 Matched Pairs)

* **Exact Raw (Name + Address) Match:** **0.00%** (0 out of 34,511 pairs)
* **Exact Raw Name Match:** **4.61%**
* **Exact Raw Address Match:** **2.36%**
* **Case-Insensitive Name Match:** **10.70%**
* **Case-Insensitive Address Match:** **7.04%**
* **Country Equality:** **100.00%** (All true matches belong to the exact same country)

---

## 3. Observed Business Name Noise Patterns

### A. Typos & Character Perturbations
* **Observed in Data:** Yes (High frequency across US and India).
* **Examples:**
  * S1: `Maure Williams Colombier Inc` $\rightarrow$ Match: `Maure Wilblims Colombier Inc`
  * S1: `Printech Solutions Pvt Ltd` $\rightarrow$ Match: `Printech Solutins Pvt Ltd`

### B. Legal Suffix Inconsistencies & Omissions
* **Observed in Data:** Yes (Extremely frequent).
* **Examples:**
  * S1: `Maure Williams Colombier Inc` $\rightarrow$ Match: `Maure Williams Colombier`
  * S1: `ABC Logistics Limited` $\rightarrow$ Match: `ABC Logistics Ltd`
  * S1: `Raj Investments LLP` $\rightarrow$ Match: `Raj Investments`

### C. Web Domain / Digital Identifier Variants
* **Observed in Data:** Yes.
* **Examples:**
  * S1: `Maure Williams Colombier Inc` $\rightarrow$ Match: `maurewilliamscolombier.com`

### D. Trade Name / Location Tag Additions
* **Observed in Data:** Yes.
* **Examples:**
  * S1: `Maure Williams Colombier Inc` $\rightarrow$ Match: `Maure Williams Inc Center`

### E. Special Character Encoding & Punctuation Noise
* **Observed in Data:** Yes (`&` vs `and`, dots in legal abbreviations, accented / non-ASCII characters).

---

## 4. Observed Address Noise Patterns

### A. Missing Address Components
* **Observed in Data:** Yes (~3.3% of S2/S3 entities have completely empty addresses).

### B. Landmark-Based Descriptions & Formatting Variations
* **Observed in Data:** Yes (Especially prevalent in India records, e.g., "Near SBI ATM", "Opposite Railway Station").

### C. Postal Code / PIN Differences & Reordering
* **Observed in Data:** Yes (PIN code appended at end vs middle, formatting as 6-digit number vs hyphenated).

---

## 5. Summary of Non-Observed Patterns / Insufficient Evidence
* **Cross-Country Matches:** Not observed. 100% of true matches in training share identical country labels.
