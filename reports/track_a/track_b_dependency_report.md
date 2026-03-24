# Track B Dependency Memo

## What Track B must keep stable

- Target name: `ARR_DEL15`
- Temporal split: train 2021-2024, test 2025
- Core metrics: ROC-AUC, PR-AUC, F1, confusion matrix
- Narrative scope: associative/descriptive + predictive, not causal

## What does not block Track A

- Track B model choice
- Track B tuning strategy
- Track B optional interpretability work

## What could block final cross-track comparison

- Different target definition
- Different test year
- Different metric definitions
- Different artifact naming that makes comparison ambiguous
