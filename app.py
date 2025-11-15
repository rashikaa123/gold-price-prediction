import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# -------------------- LOAD MODEL -----------------------
model = joblib.load("gold_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_cols = joblib.load("feature_cols.joblib")

st.title("💰 Gold Price Prediction App")

# -------------------------------------------------------
# PART 1 — SHORT TERM GOLD PREDICTION (ML MODEL)
# -------------------------------------------------------

st.header("🔍 Predict Gold Price Using ML Model (Short-Term)")
st.write("Enter financial indicators below:")

inputs = {}
for col in feature_cols:
    inputs[col] = st.number_input(f"Enter {col} value", value=0.0)

if st.button("Predict Gold Price"):
    input_df = pd.DataFrame([inputs])
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]

    st.success(f"Predicted GOLD Price (USD): {prediction:.2f}")

# -------------------------------------------------------
# PART 2 — FUTURE GOLD PRICE (CAGR METHOD)
# -------------------------------------------------------

st.header("📈 Predict Future Gold Price in INR (Long-Term)")

current_year = datetime.datetime.now().year
future_year = st.number_input("Enter future year (2030, 2045 etc.)",
                              min_value=current_year,
                              max_value=2100,
                              value=2030)

current_gold_price_inr = 65000  # per 10 grams
cagr = 0.08  # 8% yearly

years_diff = future_year - current_year
future_price = current_gold_price_inr * ((1 + cagr) ** years_diff)

if st.button("Predict Future INR Price"):
    st.success(f"Estimated Gold Price in {future_year}: ₹{future_price:,.2f} per 10 grams")
