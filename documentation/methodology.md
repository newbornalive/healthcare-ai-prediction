# Methodology

## Objective

Estimate a calibrated probability of the combined `prediabetes or diabetes` outcome from 21 survey-derived health and demographic indicators.

## Data Partitioning

The project uses a single reproducible stratified split:

- 60% training
- 10% validation
- 15% calibration and threshold selection
- 15% locked test evaluation

All transformations are fitted after splitting. The standard scaler used by Logistic Regression and the CNN is fitted only on the training partition.

## Models

### Logistic Regression

A class-weighted logistic regression provides a transparent statistical baseline. Continuous and ordinal inputs are standardised.

### Random Forest

The forest uses class-balanced bootstrap samples, constrained depth, and minimum leaf size to limit overfitting.

### XGBoost

XGBoost uses imbalance weighting, histogram tree construction, subsampling, column sampling, regularisation, and validation-based early stopping.

### 1D CNN

The CNN consists of two Conv1D layers followed by global average pooling and a linear classifier. It is trained with AdamW and class-weighted binary cross-entropy.

A fixed 40,000-record stratified training subset and 10,000-record validation subset are used for the CNN so that the deep-learning experiment is reproducible on ordinary CPU hardware. Calibration and testing still use the complete designated partitions.

The feature order is fixed and stored in model metadata. The convolutional architecture is an experimental benchmark: tabular health indicators do not possess an inherent spatial sequence.

## Probability Calibration

Every model is calibrated with Platt scaling on the separate calibration partition. The calibrator receives the log-odds of the raw predicted probability and estimates a one-variable logistic recalibration model.

## Threshold Selection

For each model, the calibration partition is searched over thresholds from 0.01 to 0.99. The selected threshold maximises specificity among thresholds reaching at least 80% sensitivity. If the target is infeasible, the maximum Youden-index threshold is used.

Threshold selection is intentionally separated from final test evaluation.

## Metrics

Probability metrics:

- ROC-AUC
- PR-AUC
- Brier score
- Log loss
- Expected calibration error

Operating-point metrics:

- Sensitivity
- Specificity
- Precision
- Negative predictive value
- F1 score
- Accuracy
- Confusion-matrix counts

Forty nonparametric bootstrap samples provide illustrative percentile intervals for ROC-AUC, PR-AUC, and Brier score.

## Explainability

- XGBoost: mean absolute SHAP values on a fixed 1,500-record test sample
- CNN: mean absolute Integrated Gradients on 500 standardised test observations

Explanations describe model behaviour. They do not establish causality or medical importance.

## Subgroup Analysis

The best model by test PR-AUC is evaluated descriptively across:

- Sex
- Broad age bands
- Income bands
- Education bands

Reported metrics include prevalence, mean predicted risk, calibration gap, ROC-AUC, PR-AUC, sensitivity, and specificity. These comparisons are monitoring diagnostics and not a complete fairness assessment.
