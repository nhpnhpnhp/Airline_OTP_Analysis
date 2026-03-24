# Track A Slide Deck Draft

## Slide 1 - Title
- Airline OTP Analysis
- Track A: Pre-flight Delay Prediction

## Slide 2 - Problem Motivation
- Why OTP matters for airlines and passengers
- Why January 2021-2025 BTS data is relevant

## Slide 3 - Current Project Scope
- Preprocessing pipeline completed
- Exploratory analysis completed
- Track A modeling added as the first predictive branch

## Slide 4 - Data Pipeline
- Raw BTS CSVs
- Clean parquet outputs
- ML-ready Track A dataset

## Slide 5 - Target and Split
- Target: ARR_DEL15
- Train: 2021-2024
- Test: 2025
- Why temporal split matters

## Slide 6 - Track A Feature Design
- Schedule/time features
- Route and airport frequency features
- Historical OTP features
- No post-departure leakage

## Slide 7 - Leakage Audit
- Forbidden columns removed
- Why Track A remains pre-flight only

## Slide 8 - Statistical Test 1
- Chi-square YEAR vs ARR_DEL15
- Key interpretation

## Slide 9 - Statistical Test 2
- Kruskal-Wallis ARR_DELAY_NEW by DEP_TIME_BLK
- Key interpretation

## Slide 10 - Comparative Association Findings
- Carrier
- Route
- Time block

## Slide 11 - Models
- Logistic Regression baseline
- Lightweight tree model

## Slide 12 - Evaluation Metrics
- ROC-AUC
- PR-AUC
- F1
- Confusion Matrix

## Slide 13 - Model Comparison
- Table of Track A results
- Selected best model: Logistic Regression

## Slide 14 - Feature Importance
- Permutation importance for the best model

## Slide 15 - Conclusion and Next Steps
- What Track A achieved
- What remains optional
- How Track B can align without blocking Track A
