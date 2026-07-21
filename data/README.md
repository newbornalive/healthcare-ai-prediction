# Data

## Source

This project uses the CDC Diabetes Health Indicators dataset distributed through the UCI Machine Learning Repository as dataset 891.

```bash
python data/download_data.py
```

The download script writes:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

The raw CSV is excluded from Git because the source dataset should be retrieved from its documented distribution point.

## Included Sample

`data/sample/diabetes_sample.csv` contains a fixed stratified sample of 1,500 rows for lightweight schema inspection and prediction demonstrations. It must not be used to reproduce the reported model metrics.

## Outcome

`Diabetes_binary` is coded as:

- `0`: no diabetes
- `1`: prediabetes or diabetes

This combined label does not distinguish between prediabetes and diagnosed diabetes.

## Citation

CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919
