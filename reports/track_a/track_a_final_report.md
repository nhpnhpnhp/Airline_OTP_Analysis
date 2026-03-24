# Track A Final Report

## 1. Executive Summary

Track A focuses on pre-flight features for predicting `ARR_DEL15` with a temporal split between 2021-2024 (train) and 2025 (test). The workflow combines compact statistical analysis, leakage-aware modeling, and final reporting artifacts suitable for the course project.

## 2. Experimental Setup

- Target: `ARR_DEL15`
- Train period: 2021-2024
- Test period: 2025
- Track A features: pre-flight and schedule-derived numeric features only
- Leakage rule: exclude operational outcome variables such as arrival outcomes, taxi-in, wheels-on, and post-arrival delay causes

## 3. Statistical Analysis

### 3.1 Chi-square Test: YEAR vs ARR_DEL15

- Statistic: 31026.8233
- Degrees of freedom: 4
- p-value: 0
- Effect size (Cramers V): 0.1129

Interpretation: OTP distribution differs across years, but the effect size should be read alongside business meaning rather than p-value alone because the sample is large.

### 3.2 Kruskal-Wallis Test: ARR_DELAY_NEW by DEP_TIME_BLK

- Statistic: 21825.1056
- Degrees of freedom: 18
- p-value: 0
- Effect size (Epsilon squared): 0.0090

Interpretation: delay magnitude differs across departure time blocks, supporting time-of-day as a relevant associative factor for Track A.

## 4. Comparative Association Analysis

### 4.1 Top Carrier Associations

| OP_CARRIER | flights | delay_rate |
| --- | --- | --- |
| WN | 474178 | 0.1799 |
| DL | 336839 | 0.1710 |
| AA | 325701 | 0.2043 |
| OO | 275984 | 0.1984 |
| UA | 237279 | 0.1942 |
| YX | 118273 | 0.1588 |
| MQ | 97683 | 0.1993 |
| B6 | 87727 | 0.2734 |
| NK | 86397 | 0.2270 |
| OH | 80343 | 0.2081 |

### 4.2 High-Delay Routes (min 500 flights)

| ROUTE | flights | delay_rate |
| --- | --- | --- |
| BOS-PBI | 784 | 0.3686 |
| ASE-DFW | 555 | 0.3658 |
| ORD-ASE | 626 | 0.3594 |
| LAX-ASE | 652 | 0.3543 |
| DFW-ASE | 550 | 0.3473 |
| DEN-ASE | 1025 | 0.3454 |
| JFK-PBI | 810 | 0.3444 |
| BOS-SJU | 633 | 0.3302 |
| ASE-ORD | 627 | 0.3301 |
| BOS-RSW | 968 | 0.3285 |

### 4.3 Time Block Summary

| DEP_TIME_BLK | delay_rate | avg_delay_new |
| --- | --- | --- |
| 2100-2159 | 0.2387 | 18.7427 |
| 1900-1959 | 0.2482 | 18.7070 |
| 1800-1859 | 0.2386 | 17.6625 |
| 2200-2259 | 0.2199 | 17.5582 |
| 2000-2059 | 0.2404 | 17.5074 |
| 1700-1759 | 0.2288 | 16.9010 |
| 1600-1659 | 0.2248 | 16.3848 |
| 1500-1559 | 0.2193 | 16.1138 |
| 1400-1459 | 0.2072 | 15.3326 |
| 1300-1359 | 0.2033 | 15.1525 |

These findings are descriptive associations. They support feature relevance, but they are not presented as causal driver analysis.

## 5. Leakage Audit and Temporal Split

- The preprocessing stage already removed forbidden leakage columns for Track A.
- Temporal evaluation uses 2025 as a forward-looking test set, which is more realistic than a random split for OTP prediction.
- The Track A feature set remains aligned with pre-flight availability assumptions.

## 6. Track A Modeling

### 6.1 Model Comparison

| model | roc_auc | pr_auc | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6099 | 0.2633 | 0.2310 | 0.6657 | 0.3430 | 0.4400 | 65329 | 206655 | 217484 | 32801 |
| Decision Tree | 0.5945 | 0.2278 | 0.2096 | 0.7354 | 0.3263 | 0.2400 | 72168 | 152072 | 272067 | 25962 |

### 6.2 Best Model

The selected best model is **Logistic Regression** based on test-set discrimination and overall balance between ROC-AUC, PR-AUC, and F1.

## 7. Permutation Importance

| feature | importance_drop_auc |
| --- | --- |
| DAY_OF_MONTH | 0.0351 |
| ORIGIN_HIST_OTP | 0.0142 |
| CARRIER_HIST_OTP | 0.0039 |
| CRS_DEP_SIN | 0.0029 |
| CRS_ARR_SIN | 0.0020 |
| IS_WEEKEND | 0.0010 |
| CRS_DEP_TIME_MIN | 0.0009 |
| ROUTE_FREQ | 0.0003 |
| DEST_FREQ | 0.0003 |
| CRS_DEP_COS | 0.0002 |

Permutation importance was computed only for the selected best model to keep interpretability focused and within scope.

## 8. Track B Dependency Note

Track A does not depend on Track B implementation to finish its own modeling. The only required coordination point is keeping the same target name, temporal split, and core evaluation metrics for final cross-track comparison.

## 9. Limitations and Next Steps

- Track A uses only pre-flight information, so there is an upper limit on achievable performance.
- The tree model is intentionally lightweight to keep project scope realistic.
- Optional future work: SHAP for one boosting-style model, lightweight dashboard overview, and a side-by-side comparison once Track B is finalized.
