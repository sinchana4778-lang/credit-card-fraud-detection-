import streamlit as st
import numpy as np
import joblib
import os

# =========================
# LOAD MODEL + SCALER
# =========================

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it's Fraud or Normal")

# =========================
# INPUT FORM
# =========================

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=100.0)

st.subheader("Enter V1 - V28 values")

v = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    v.append(val)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Check Fraud Risk"):
    
    # Create input array
    features = np.array([[time] + v + [amount]])
    
    # Scale input
    features_scaled = scaler.transform(features)
    
    # Predict probability
    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader("🔍 Result")

    st.write(f"Fraud Probability: **{prob:.4f}**")

    # Risk logic
    if prob > 0.7:
        st.error("🚨 HIGH RISK FRAUD")
    elif prob > 0.4:
        st.warning("⚠️ MEDIUM RISK TRANSACTION")
    else:
        st.success("✅ LOW RISK (SAFE TRANSACTION)")