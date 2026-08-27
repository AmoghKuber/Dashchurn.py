# Telecom Churn Prediction - Streamlit Dashboard

## Files required

Keep these files in the same GitHub repository:

- `app.py`
- `requirements.txt`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push all three files to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file to `app.py`.
5. Deploy.

The dashboard contains:

- Overview
- Customer Churn Prediction
- Model Performance
- Churn Analysis
- SHAP Explainability

The model follows the supplied project structure: categorical encoding, MinMax scaling of tenure/MonthlyCharges/TotalCharges, removal of the six low-relation features, SMOTE balancing, and a stacking classifier using XGBoost, LightGBM, Random Forest and Decision Tree.
