import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details below to predict whether they will churn or stay.")

# ── Input fields ─────────────────────────────────────────────
st.subheader("Customer Details")

credit_score    = st.number_input("Credit Score",        min_value=300,  max_value=850,  value=600)
gender          = st.selectbox("Gender",                 ["Male", "Female"])
age             = st.number_input("Age",                 min_value=18,   max_value=100,  value=35)
tenure          = st.number_input("Tenure (years)",      min_value=0,    max_value=10,   value=5)
balance         = st.number_input("Balance",             min_value=0.0,                  value=50000.0)
num_of_products = st.number_input("Number of Products",  min_value=1,    max_value=4,    value=1)
has_cr_card     = st.selectbox("Has Credit Card",        ["Yes", "No"])
is_active       = st.selectbox("Is Active Member",       ["Yes", "No"])
estimated_salary= st.number_input("Estimated Salary",    min_value=0.0,                  value=50000.0)
geography       = st.selectbox("Geography",              ["France", "Germany", "Spain"])

# ── Preprocess input ─────────────────────────────────────────
gender_encoded    = 1 if gender == "Male" else 0
has_cr_card_enc   = 1 if has_cr_card == "Yes" else 0
is_active_enc     = 1 if is_active == "Yes" else 0

geography_france  = 1 if geography == "France"  else 0
geography_germany = 1 if geography == "Germany" else 0
geography_spain   = 1 if geography == "Spain"   else 0

# ── Predict button ───────────────────────────────────────────
if st.button("Predict"):
    input_data = np.array([[credit_score, gender_encoded, age, tenure,
                            balance, num_of_products, has_cr_card_enc,
                            is_active_enc, estimated_salary,
                            geography_france, geography_germany, geography_spain]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN — Probability: {round(probability*100, 2)}%")
    else:
        st.success(f"✅ This customer is likely to STAY — Probability: {round((1-probability)*100, 2)}%")