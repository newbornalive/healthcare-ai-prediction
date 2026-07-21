# Model Card

## Model Family

Diabetes risk prediction benchmark comparing Logistic Regression, Random Forest, XGBoost, and an experimental PyTorch 1D CNN.

## Intended Use

- Education and portfolio demonstration
- Reproducible comparison of tabular machine-learning methods
- Research into discrimination, calibration, threshold sensitivity, and subgroup stability

## Out-of-Scope Use

The models must not be used to:

- Diagnose diabetes or prediabetes
- Recommend treatment
- Replace laboratory testing or clinical assessment
- Determine insurance, employment, credit, benefits, or healthcare eligibility
- Rank or penalise individual patients

## Training Data

CDC Diabetes Health Indicators dataset distributed through UCI. The outcome is a combined indicator of prediabetes or diabetes. Inputs include health conditions, health behaviours, general-health ratings, functional limitations, sex, age, education, and income.

## Test Performance

| Model | ROC-AUC | PR-AUC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.828 | 0.423 | 0.097 | 0.803 | 0.705 |
| Random Forest | 0.824 | 0.414 | 0.098 | 0.813 | 0.690 |
| Logistic Regression | 0.820 | 0.391 | 0.100 | 0.817 | 0.678 |
| 1D CNN | 0.715 | 0.286 | 0.111 | 0.790 | 0.507 |

The selected XGBoost threshold is 0.14. At that illustrative operating point, approximately 80.3% of positive outcomes are detected and 70.5% of negative outcomes are correctly screened out.

## Principal Findings

- XGBoost ranked outcomes most effectively and produced the lowest Brier score.
- Tree-based models outperformed the CNN on this tabular dataset.
- Calibration gaps were small overall after Platt scaling.
- A common global threshold produced materially different sensitivity and specificity across age, income, and education groups.

## Ethical and Reliability Risks

- Survey responses may contain reporting and recall error.
- Income and education may proxy structural disadvantage.
- The combined target obscures important differences between prediabetes and diabetes.
- The dataset is not a contemporary clinical validation cohort.
- Performance differences across subgroups may be caused by prevalence, feature distribution, label quality, sample size, or model misspecification.
- High sensitivity requires accepting a substantial false-positive rate.

## Monitoring Requirements for Any Future Research Extension

- External validation
- Temporal and geographic drift analysis
- Calibration monitoring
- Subgroup sample-size and uncertainty reporting
- Clinical review of variables and thresholds
- Documented human oversight
