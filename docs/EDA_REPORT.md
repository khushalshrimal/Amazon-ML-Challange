# Dataset EDA & Profiling Report — Amazon ML Challenge 2026

## 1. Executive Summary
This report summarizes empirical profiling performed on the training datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`).

## 2. Source-by-Source Data Characteristics

### Source 1 (`train_source1.tsv`) — Deduplicated Reference Source
* **Total Records:** 2,206,821
* **Unique `entity_id` Count:** 2,206,821 (100% unique)
* **Missing Fields:**
  * `business_name`: 0 (0.00%)
  * `business_address`: 0 (0.00%)
  * `country`: 0 (0.00%)
* **Uniqueness & Duplication:**
  * Unique Business Names: 1,539,229 (845,385 rows share a name with another S1 record)
  * Unique Business Addresses: 2,130,606 (116,304 rows share an address with another S1 record)
  * Unique `(business_name, business_address)` pairs: 2,206,821 (0 duplicates in S1 reference pairs)
* **Country Breakdown:**
  * `US`: 1,323,633 (60.00%)
  * `India`: 883,188 (40.00%)

### Source 2 (`train_source2.tsv`) — Potential Matching Source
* **Total Records:** 5,034,616
* **Unique `entity_id` Count:** 5,034,616 (100% unique)
* **Missing Fields:**
  * `business_name`: 2 (< 0.01%)
  * `business_address`: 168,967 (3.36%)
  * `country`: 0 (0.00%)
* **Uniqueness & Duplication:**
  * Unique Business Names: 4,402,008
  * Unique Business Addresses: 4,337,261
  * Unique `(business_name, business_address)` pairs: 5,008,725 (50,969 duplicate name-address pairs)
* **Country Breakdown:**
  * `US`: 3,016,817 (59.92%)
  * `India`: 2,017,799 (40.08%)

### Source 3 (`train_source3.tsv`) — Potential Matching Source
* **Total Records:** 5,285,603
* **Unique `entity_id` Count:** 5,285,603 (100% unique)
* **Missing Fields:**
  * `business_name`: 13 (< 0.01%)
  * `business_address`: 175,916 (3.33%)
  * `country`: 0 (0.00%)
* **Uniqueness & Duplication:**
  * Unique Business Names: 4,651,608
  * Unique Business Addresses: 4,632,764
  * Unique `(business_name, business_address)` pairs: 5,266,722 (37,283 duplicate name-address pairs)
* **Country Breakdown:**
  * `US`: 3,170,056 (59.98%)
  * `India`: 2,115,547 (40.02%)

---

## 3. Ground Truth Matching Analysis (`train_ground_truth.tsv`)

* **Total S1 Entities Evaluated:** 2,206,821
* **Singletons (0 matches):** 123,247 (5.58%)
* **1 Match:** 119,157 (5.40%)
* **Multi-Matches ($\ge 2$ matches):** 1,964,417 (89.01%)
* **Maximum Matched Entities for Single S1:** 11 matches

### Match Cardinality Distribution Table

| Matched Entities Count | Number of S1 Entities | Percentage (%) | Cum. Percentage (%) |
| :---: | :---: | :---: | :---: |
| **0 (Singleton)** | 123,247 | 5.58% | 5.58% |
| **1** | 119,157 | 5.40% | 10.98% |
| **2** | 375,212 | 17.00% | 27.98% |
| **3** | 530,841 | 24.05% | 52.03% |
| **4** | 484,115 | 21.94% | 73.97% |
| **5** | 321,957 | 14.59% | 88.56% |
| **6** | 164,868 | 7.47% | 96.03% |
| **7** | 63,968 | 2.90% | 98.93% |
| **8** | 18,680 | 0.85% | 99.78% |
| **9** | 4,205 | 0.19% | 99.97% |
| **10** | 534 | 0.02% | 99.99% |
| **11** | 37 | < 0.01% | 100.00% |

---

## 4. Key EDA Insights for Candidate Blocking & Feature Modeling
1. **High Multi-Match Rate:** 89.01% of Source 1 reference entities have multiple matching records across Source 2 and Source 3 (most commonly 3 or 4 matches).
2. **Singleton Handling:** 5.58% of reference entities have no matches. Under $F_{0.5}$, predicting an empty set for these entities earns full credit (1.0).
3. **Missing Addresses in S2 & S3:** ~3.3% of S2 and S3 records have missing address values. Candidate generation and feature scoring must handle missing address fields gracefully without failing.
4. **Name & Address Duplication:** High repetition of business names (e.g. legal entity names or common trade names) occurs across sources. Address uniqueness is high (~96%+), making address tokens strong blocking keys.
