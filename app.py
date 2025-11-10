import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Housing Loan Approval", layout="centered")

# Load your saved model (must be present in the same folder)
model, feature_cols = load("dt_housing_model.joblib")

st.title("🏦 Housing Finance Loan Approval Prediction")
st.write("Enter values to check whether the loan should be approved.")

# Input fields
FOIR = st.number_input("FOIR (Fixed Obligations To Income Ratio)", min_value=0.0, step=0.1)
LTV  = st.number_input("LTV (Loan to Value Ratio)", min_value=0.0, step=0.1)
IAR  = st.number_input("IAR (Interest to Asset Ratio)", min_value=0.0, step=0.1)

if st.button("Predict Loan Decision"):
    # Create a row with default values
    row = {col: 0 for col in feature_cols}

    if "FOIR" in row: row["FOIR"] = FOIR
    if "LTV"  in row: row["LTV"]  = LTV
    if "IAR"  in row: row["IAR"]  = IAR

    X_input = pd.DataFrame([row], columns=feature_cols)

    pred = int(model.predict(X_input)[0])

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
