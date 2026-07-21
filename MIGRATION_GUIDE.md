# Migration Guide

Replace the contents of the existing `healthcare-ai-prediction` repository with the contents of this project folder.

## Remove from the Existing Repository

- `data/dummy.csv`
- the empty `notebooks/exploratory_analysis.ipynb`
- `scripts/Tableau.py`
- the empty `scripts/train_model.py`
- `Results/Results.txt`
- the previous README and empty requirements file

The hospital-visit forecasting code is a different analytical problem and should not remain in this diabetes risk classification repository.

## Expected Root Structure

```text
README.md
data/
documentation/
images/
models/
notebooks/
results/
scripts/
src/
tests/
```

## Suggested Commit Summary

```text
Rebuild healthcare AI project with calibrated ML and CNN evaluation
```

## Suggested GitHub About

```text
Diabetes risk prediction using XGBoost, Random Forest, Logistic Regression, and a PyTorch 1D CNN with calibration and subgroup evaluation.
```
