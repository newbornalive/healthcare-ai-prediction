# Responsible Use and Limitations

## Risk Prediction Is Not Diagnosis

The target is based on an analytical survey label. A predicted probability does not establish disease status. Clinical diagnosis requires appropriate medical assessment and testing.

## Sensitive and Socioeconomic Variables

The dataset contains sex, income, and education. These variables may improve statistical prediction while also reflecting unequal access to healthcare, employment conditions, neighbourhood resources, and other structural factors.

Their inclusion in this research benchmark does not imply that they are appropriate inputs for a deployed medical system.

## Subgroup Results

The subgroup table demonstrates that a single global threshold does not produce identical operating characteristics across all populations. For example, sensitivity is lower among the youngest age band because positive outcomes are relatively rare and risk distributions differ.

No subgroup gap should be interpreted as proof of discrimination or fairness without:

- confidence intervals
- error and label-quality assessment
- clinical context
- representation analysis
- external validation
- investigation of alternative thresholds and modelling choices

## Interpretability

SHAP and Integrated Gradients explain the fitted model's mathematical response to features. They do not demonstrate that changing a feature would change diabetes risk, nor that the feature is clinically causal.

## Deployment

A real healthcare deployment would require, at minimum:

- clearly defined clinical use case
- prospective validation
- privacy and security review
- clinician and patient input
- regulatory and legal assessment
- workflow evaluation
- monitoring for calibration and population drift
- procedures for human review and contestability
