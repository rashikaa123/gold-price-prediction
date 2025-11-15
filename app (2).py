import streamlit as st
import joblib
import datetime

# Load ML Model
model = joblib.load("gold_year_model.joblib")

st.title("💰 Gold Price Predictor")

st.write("Enter any year to predict Gold Price in INR")

current_year = datetime.datetime.now().year

year = st.number_input("Enter Year", min_value=1900, max_value=2100, value=current_year)

if st.button("Predict"):
    price = model.predict([[year]])[0]
    st.success(f"Estimated Gold Price in {year}: ₹{price:,.2f} per 10 grams")
