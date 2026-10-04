import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

MODEL_PATH = "model/fraud_model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

except Exception as e:
    st.error("❌ Model or Preprocessor could not be loaded.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Enter transaction details to check whether the transaction is fraudulent."
)

st.divider()


# --------------------------------------------------
# Input Fields
# --------------------------------------------------

transaction_amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=10.0
)

transaction_hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)

merchant_category = st.text_input(
    "Merchant Category",
    value="retail"
)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    options=[0, 1]
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    options=[0, 1]
)

device_trust_score = st.number_input(
    "Device Trust Score",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=2,
    step=1
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🔍 Check Transaction"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "amount": [transaction_amount],
        "transaction_hour": [transaction_hour],
        "merchant_category": [merchant_category],
        "foreign_transaction": [foreign_transaction],
        "cardholder_age": [cardholder_age],
        "location_mismatch": [location_mismatch],
        "device_trust_score": [device_trust_score],
        "velocity_last_24h": [velocity_last_24h]
    })

    try:

        # Preprocess input
        processed_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Get probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_data)[0][1] * 100
        else:
            probability = 100.0 if prediction == 1 else 0.0

        st.divider()

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ TRANSACTION APPEARS TO BE LEGITIMATE")

        # Fraud probability
        st.metric(
            "Fraud Probability",
            f"{probability:.2f}%"
        )

        # --------------------------------------------------
        # Risk Level
        # --------------------------------------------------

        if probability >= 70:
            st.error("🔴 Risk Level: HIGH")

        elif probability >= 40:
            st.warning("🟠 Risk Level: MEDIUM")

        else:
            st.info("🟡 Risk Level: LOW")


    except Exception as e:
            st.exception(e)