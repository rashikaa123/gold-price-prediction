import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import tempfile
from sklearn.metrics import r2_score, mean_squared_error

# ------------------ Page config --------------------
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Prediction — Year-based")

st.markdown("Predict gold price trends using historical data (India).")

# ------------------ Load data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gold_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df = df.rename(columns={'Close': 'Gold_Price'})
    df = df.groupby('Year')['Gold_Price'].mean().reset_index()
    df = df.sort_values('Year').reset_index(drop=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load("gold_model_new.joblib")

try:
    df = load_data()
    model = load_model()
except:
    st.error("❌ Missing dataset or model file.")
    st.stop()

# ------------------ Sidebar --------------------
st.sidebar.header("Settings")

current_year = datetime.datetime.now().year
min_year = int(df['Year'].min())
max_year = max(current_year, int(df['Year'].max()) + 10)

target_year = st.sidebar.number_input("Target Year", min_value=1900, max_value=2100, value=current_year)
horizon = st.sidebar.slider("Future Range", min_value=int(df['Year'].max()), max_value=max_year, value=max(target_year, int(df['Year'].max()) + 5))

show_table = st.sidebar.checkbox("Show Dataset", True)
compare_mode = st.sidebar.checkbox("Compare Years")
download_pdf = st.sidebar.checkbox("Enable PDF Report")

# ------------------ Prediction function --------------------
def predict_years(years):
    return model.predict(np.array(years).reshape(-1, 1))

# ------------------ Build predictions --------------------
hist_years = df['Year'].tolist()
plot_years = list(range(min_year, horizon + 1))

# vectorized prediction (FAST)
all_preds = predict_years(plot_years)

pred_prices = []
for i, y in enumerate(plot_years):
    if y in hist_years:
        val = df.loc[df['Year'] == y, 'Gold_Price'].values[0]
    else:
        val = all_preds[i]
    pred_prices.append(float(val))

# ------------------ Layout --------------------
left, right = st.columns([1, 2])

# ------------------ LEFT --------------------
with left:
    st.subheader("🔢 Predict Year")

    input_year = st.number_input("Enter Year", min_value=min_year, max_value=2100, value=target_year)

    if st.button("Predict"):
        pred = float(predict_years([input_year])[0])

        q1, q2 = np.percentile(df['Gold_Price'], [33, 66])

        if pred <= q1:
            label, color = "Low", "green"
        elif pred <= q2:
            label, color = "Medium", "orange"
        else:
            label, color = "High", "red"

        st.markdown(f"### 🎯 ₹{pred:,.2f} per 10g")
        st.markdown(f"**Category:** :{color}[{label}]")

    st.markdown("---")

    # Compare Mode
    if compare_mode:
        st.subheader("🔎 Compare Years")

        y1 = st.number_input("Year 1", value=min_year + 1)
        y2 = st.number_input("Year 2", value=min_year + 2)

        if st.button("Compare"):
            p1 = float(predict_years([y1])[0])
            p2 = float(predict_years([y2])[0])

            diff = p2 - p1
            pct = (diff / p1) * 100 if p1 != 0 else 0

            st.write(f"{y1}: ₹{p1:,.2f}")
            st.write(f"{y2}: ₹{p2:,.2f}")
            st.write(f"Difference: ₹{diff:,.2f} ({pct:.2f}%)")

    st.markdown("---")

    # Model Info
    st.subheader("📊 Model Info")

    y_pred = model.predict(df[['Year']])
    r2 = r2_score(df['Gold_Price'], y_pred)
    rmse = np.sqrt(mean_squared_error(df['Gold_Price'], y_pred))

    st.write(f"Model: {type(model).__name__}")
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"RMSE: {rmse:.2f}")

# ------------------ RIGHT --------------------
with right:
    st.subheader("📈 Trends")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df['Year'], df['Gold_Price'], marker='o', label='Historical')
    ax.plot(plot_years, pred_prices, linestyle='--', label='Prediction')

    if input_year in plot_years:
        val = float(predict_years([input_year])[0])
        ax.scatter(input_year, val, s=100)

    ax.set_xlabel("Year")
    ax.set_ylabel("Gold Price (₹)")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    if show_table:
        st.subheader("Dataset")
        st.dataframe(df)

# ------------------ PDF --------------------
def create_pdf(year):
    pred = float(predict_years([year])[0])

    y_pred = model.predict(df[['Year']])
    r2 = r2_score(df['Gold_Price'], y_pred)
    rmse = np.sqrt(mean_squared_error(df['Gold_Price'], y_pred))

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Gold Price Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Year: {year}", ln=True)
    pdf.cell(0, 8, f"Predicted Price: ₹{pred:,.2f}", ln=True)
    pdf.cell(0, 8, f"R2 Score: {r2:.4f}", ln=True)
    pdf.cell(0, 8, f"RMSE: {rmse:.2f}", ln=True)

    # Save plot temp
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name)

    pdf.image(tmp.name, w=180)

    return pdf.output(dest="S").encode("latin-1")

# ------------------ Download --------------------
if download_pdf:
    st.sidebar.markdown("---")
    pdf_year = st.sidebar.number_input("PDF Year", value=current_year)

    if st.sidebar.button("Generate PDF"):
        pdf_bytes = create_pdf(int(pdf_year))
        st.sidebar.download_button("Download", pdf_bytes, file_name="report.pdf")

# ------------------ Footer --------------------
st.markdown("---")
st.info("⚠️ Predictions are trend-based and not guaranteed for long-term future.")