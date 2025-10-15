import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Load Pretrained Model
# =========================
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =========================
# 2️⃣ Set Up Page
# =========================
st.set_page_config(
    page_title="Credit Risk Prediction App",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Explainable Credit Risk Prediction")
st.markdown("""
This dashboard predicts **credit risk** for loan applicants using a trained machine learning model (XGBoost),  
and explains predictions using **SHAP (SHapley Additive exPlanations)**.
""")

# =========================
# 3️⃣ Define Feature Inputs
# =========================

# Replace with your actual feature names
categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

numerical_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

# Sidebar inputs
st.sidebar.header("📌 Applicant Information")

def user_input_features():
    person_age = st.sidebar.slider("Age", 18, 100, 35)
    person_income = st.sidebar.number_input("Annual Income", min_value=1000, max_value=500000, value=50000)
    person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.sidebar.slider("Employment Length (years)", 0, 50, 5)
    loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.sidebar.number_input("Loan Amount", min_value=500, max_value=50000, value=10000)
    loan_int_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 12.0)
    loan_percent_income = st.sidebar.slider("Loan Percent Income", 0.0, 1.0, 0.2)
    cb_person_default_on_file = st.sidebar.selectbox("Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (years)", 0, 50, 10)

    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# =========================
# 4️⃣ Preprocess Inputs
# =========================
# 👉 You should use the same preprocessing pipeline as during training
# (e.g., Label Encoding for categorical variables, Scaling for numeric)

# For demo: simple encoding (you should replace this with your fitted encoders)
def preprocess(df):
    df_encoded = df.copy()
    # Basic encoding for categorical
    for col in categorical_features:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes
    return df_encoded

processed_input = preprocess(input_df)

# =========================
# 5️⃣ Make Prediction
# =========================
prediction = model.predict(processed_input)[0]
prediction_proba = model.predict_proba(processed_input)[0][1]

st.subheader("🔮 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Default — Probability: {prediction_proba:.2%}")
else:
    st.success(f"✅ Low Risk of Default — Probability: {prediction_proba:.2%}")

# =========================
# 6️⃣ SHAP Explanations
# =========================
st.subheader("📈 SHAP Explanations")

# Initialize SHAP Explainer once
@st.cache_resource
def get_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer

explainer = get_explainer(model)
shap_values = explainer(processed_input)

# Individual Waterfall Plot
st.markdown("**Individual Prediction Explanation**")
fig, ax = plt.subplots(figsize=(8, 5))
shap.waterfall_plot(shap_values[0], show=False)
st.pyplot(fig)

# Global Summary Plot (optional)
st.markdown("**Global Feature Importance (Summary)**")
# Use a small background sample for performance if needed
background_sample = processed_input  # in practice, use a sample of training data
shap_values_global = explainer(background_sample)

fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values_global.values, background_sample, plot_type="bar", show=False)
st.pyplot(fig_summary)

# =========================
# 7️⃣ Footer
# =========================
st.markdown("""
---
📝 **Note:** This is a demo dashboard for educational purposes.  
Model: XGBoost | Explanations: SHAP  
© 2025 Capstone Project
""")
