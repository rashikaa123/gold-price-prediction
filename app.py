import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature columns
model = joblib.load("gold_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_cols = joblib.load("feature_cols.joblib")

# Load dataset for feature engineering (same CSV you used)
df = pd.read_csv("gld_price_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# Feature engineering (must match training)
numeric_cols = ['SPX','GLD','USO','SLV','EUR/USD']
lags = [1,2,3]
for col in numeric_cols:
    for l in lags:
        df[f"{col}_lag_{l}"] = df[col].shift(l)
    df[f"{col}_rmean_7"] = df[col].rolling(window=7).mean()
    df[f"{col}_rstd_7"] = df[col].rolling(window=7).std()

df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df = df.dropna().reset_index(drop=True)

st.title("💰 Gold Price Prediction")

st.write("Enter today's market values (SPX, USO, SLV, EUR/USD) and click Predict.")

SPX_val = st.number_input("SPX value", value=float(df['SPX'].iloc[-1]))
USO_val = st.number_input("USO value", value=float(df['USO'].iloc[-1]))
SLV_val = st.number_input("SLV value", value=float(df['SLV'].iloc[-1]))
EUR_val = st.number_input("EUR/USD value", value=float(df['EUR/USD'].iloc[-1]))

last = df[feature_cols].tail(1).copy()
last['SPX'] = SPX_val
last['USO'] = USO_val
last['SLV'] = SLV_val
last['EUR/USD'] = EUR_val

if st.button("Predict"):
    # Determine if model expects scaled input (LinearRegression / SVR) by checking class name
    cls_name = model.__class__.__name__
    if cls_name in ['LinearRegression', 'SVR']:
        sample_scaled = scaler.transform(last.values)
        pred = model.predict(sample_scaled)
    else:
        pred = model.predict(last.values)
    st.success(f"Predicted GOLD PRICE = {float(pred):.2f}")
