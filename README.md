# PredictAI — Intelligent Predictive Maintenance

An end-to-end machine learning application for estimating
industrial equipment failure risk.

## Features

- Random Forest failure-risk classification
- Interactive machine sensor controls
- Machine health monitoring dashboard
- Custom decision threshold
- Classification performance evaluation
- Interactive Plotly risk visualization

## Technology Stack

Python, Pandas, NumPy, scikit-learn, Streamlit,
Plotly, joblib

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset
10,000 synthetic industrial machine-operation records.

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Model Results

Evaluation on 2,000 held-out benchmark records:

| Metric | Value |
|---|---|
| Recall | 73.53% |
| Precision | 57.47% |
| F1 Score | 0.6452 |
| Failures Detected | 50 / 68 |
| Decision Threshold | 0.30 |

The threshold was selected on validation data. The test
set has been inspected during development, so these results
should be considered exploratory rather than a fully
independent final benchmark.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Dataset contains synthetic rather than real factory data.
- Estimated risks are not calibrated real-world probabilities.
- No real-time sensor integration or temporal failure prediction.
- Not validated for safety-critical industrial operations.

## Future Improvements

- Cross-validation and independent benchmark evaluation
- Explainable AI with SHAP
- Batch failure-risk prediction
- Model monitoring and drift detection
- Docker deployment
