import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Housing Loan Approval", layout="centered")

model, feature_cols = load("dt_housing_model.joblib")

st.title("🏦 Housing Finance Loan Approval Prediction")
st.write("Fill the inputs below to predict whether loan should be approved.")

FOIR   = st.number_input("FOIR", min_value=0.0, step=0.1)
MarVal = st.number_input("Market Value of Property", min_value=0.0, step=10000.0)
YrsJob = st.number_input("Years in Job", min_value=0, max_value=60, step=1)

if st.button("Predict Loan Decision"):

    row = {col: 0 for col in feature_cols}
    if "FOIR"   in row: row["FOIR"] = FOIR
    if "MarVal" in row: row["MarVal"] = MarVal
    if "YrsJob" in row: row["YrsJob"] = YrsJob

    X_input = pd.DataFrame([row], columns=feature_cols)
    pred = int(model.predict(X_input)[0])

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
