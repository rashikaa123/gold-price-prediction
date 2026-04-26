import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import yfinance as yf

# ---------------- UI STYLE ----------------
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #f5c542;
}
.stButton>button {
    background-color: #f5c542;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("💰 Gold Price Prediction App")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("gold_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df = df.rename(columns={'Close': 'Gold_Price'})
    df = df.groupby('Year')['Gold_Price'].mean().reset_index()
    df = df.sort_values('Year')
    return df

@st.cache_resource
def load_model():
    return joblib.load("gold_model_new.joblib")

df = load_data()
model = load_model()

# ---------------- LIVE GOLD PRICE ----------------
def get_live_gold_price():
    gold = yf.Ticker("GC=F")
    data = gold.history(period="1d")
    return float(data['Close'].iloc[-1])

try:
    live_price = get_live_gold_price()
    st.success(f"💰 Live Gold Price (approx): {live_price:,.2f} USD")
except:
    st.warning("Live price not available")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Options")

current_year = datetime.datetime.now().year
min_year = int(df['Year'].min())

input_year = st.sidebar.number_input("Enter Year", min_value=min_year, max_value=2100, value=current_year)

# ---------------- PREDICTION ----------------
def predict_year(year):
    return float(model.predict([[year]])[0])

pred_val = predict_year(input_year)

# Category
q1, q2 = np.percentile(df['Gold_Price'], [33, 66])
if pred_val <= q1:
    label = "Low"
elif pred_val <= q2:
    label = "Medium"
else:
    label = "High"

# ---------------- METRICS ----------------
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)
col1.metric("Predicted Price", f"{pred_val:,.2f}")
col2.metric("Category", label)

# ---------------- GRAPH ----------------
st.subheader("📈 Gold Price Trend")

plot_years = list(range(min_year, input_year + 5))
pred_prices = [predict_year(y) for y in plot_years]

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Year'], df['Gold_Price'], label='Historical', linewidth=2)
ax.plot(plot_years, pred_prices, linestyle='--', label='Prediction')

ax.set_title("Gold Price Trend")
ax.set_xlabel("Year")
ax.set_ylabel("Price")
ax.legend()
ax.grid()

st.pyplot(fig)

# ---------------- DATA TABLE ----------------
st.subheader("📚 Dataset")
st.dataframe(df)

# ---------------- PDF REPORT ----------------
def create_pdf():
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Gold Price Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Year: {input_year}", ln=True)
    pdf.cell(0, 8, f"Predicted Price: {pred_val:,.2f}", ln=True)
    pdf.cell(0, 8, f"Category: {label}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "Generated using Machine Learning Model", ln=True)
    pdf.cell(0, 8, "Project by Rashika", ln=True)

    img_path = "plot.png"
    with open(img_path, "wb") as f:
        f.write(buf.getbuffer())

    pdf.image(img_path, w=180)

    return pdf.output(dest="S").encode("latin-1")

# ---------------- DOWNLOAD BUTTON ----------------
st.sidebar.subheader("📄 Download Report")
if st.sidebar.button("Generate PDF"):
    pdf = create_pdf()
    st.sidebar.download_button("Download", pdf, "report.pdf")

# ---------------- FOOTER ----------------
st.markdown("---")
st.info("This app uses Machine Learning to predict gold price trends.")