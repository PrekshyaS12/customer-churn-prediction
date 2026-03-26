# Customer Churn Prediction

Predicting which bank customers are at risk of leaving — so the business can act before it's too late.

## Business Problem

Customer churn is one of the most expensive problems in banking. Acquiring a new customer costs 5–7x more than retaining an existing one. Yet most banks only find out a customer has left after they're already gone.

This project builds a machine learning model that identifies at-risk customers **before** they churn — giving the bank a window to intervene with targeted offers, better service, or personalised outreach.

---

## Business Impact

- Early identification of at-risk customers
- Enables proactive retention strategies
- Reduces revenue loss from unexpected churn
- Helps marketing teams prioritise high-risk segments

---

## Key Findings

- Churn rate is ~20% — 1 in 5 customers leaves the bank
- Germany has significantly higher churn than France and Spain
- Female customers churn more than male customers
- Older customers (40–60) are the highest risk age group
- Customers with only 1 product are far more likely to leave
- Inactive members churn at a much higher rate

---

## Model Performance

| Model | ROC AUC |
|---|---|
| Logistic Regression | 89.54% |
| Decision Tree | 80.79% |
| Random Forest | 94.48% |
| Gradient Boosting | 93.96% |

Random Forest was selected as the final model due to its highest ROC AUC score.

---

## Project Structure
```
customer-churn-prediction/
├── notebook.ipynb        ← full analysis, EDA and model training
├── app.py                ← streamlit web app
├── churn_model.pkl       ← saved Random Forest model
├── scaler.pkl            ← saved StandardScaler
├── churn.csv             ← dataset
└── README.md
```

---

## Run The App
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

Python · Pandas · Scikit-learn · Imbalanced-learn · Streamlit · Matplotlib · Seaborn · Pickle

---

## Limitations

- Model trained on a single bank's data — may not generalise to all banking contexts
- Does not account for external factors like economic conditions or competitor offers

---

## Future Improvements

- Try XGBoost or LightGBM for potentially better performance
- Add feature importance visualisation to explain predictions
- Build a full dashboard showing churn trends over time
- Connect to a live database for real-time predictions