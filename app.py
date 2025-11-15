# app.py — Upgraded Gold Price App (Year-based model + visuals + PDF report)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# ------------------ Page config / UI --------------------
st.set_page_config(page_title="Gold Price Predictor (Year)", layout="wide", initial_sidebar_state="expanded")
st.title("💰 Gold Price Prediction (India) — Year-based")

st.markdown(
    """
This app predicts gold price (INR per 10 grams) for any year using a trend-based ML model.
- Enter a year (past/present/future) to get prediction.
- View historical data, compare years, download a PDF report, and explore ML info.
"""
)

# ------------------ Load dataset & model --------------------
@st.cache_data
def load_data(csv_path="year_gold_data.csv"):
    df = pd.read_csv(csv_path)
    # Ensure Year column numeric
    df['Year'] = df['Year'].astype(int)
    df = df.sort_values('Year').reset_index(drop=True)
    return df

@st.cache_resource
def load_model(path="gold_year_model.joblib"):
    return joblib.load(path)

try:
    df = load_data("year_gold_data.csv")
except Exception as e:
    st.error("Could not load dataset 'year_gold_data.csv'. Make sure the file exists in the repo root.")
    st.stop()

try:
    model = load_model("gold_year_model.joblib")
except Exception as e:
    st.error("Could not load model 'gold_year_model.joblib'. Make sure the file exists in the repo root.")
    st.stop()

# ------------------ Sidebar: options --------------------
st.sidebar.header("Settings & Options")
current_year = datetime.datetime.now().year
min_year = int(df['Year'].min())
max_year = max(current_year, int(df['Year'].max()) + 10)

target_year = st.sidebar.number_input("Enter target year", min_value=1900, max_value=2100, value=current_year)
horizon = st.sidebar.slider("Plot future up to year", min_value=int(df['Year'].max()), max_value=max_year, value=max(target_year, int(df['Year'].max())+5))
show_table = st.sidebar.checkbox("Show dataset table", value=True)
compare_mode = st.sidebar.checkbox("Enable Compare Years", value=False)
download_pdf = st.sidebar.checkbox("Show PDF report button", value=True)

# ------------------ Helper: predict a list of years --------------------
def predict_years(year_list):
    arr = np.array(year_list).reshape(-1, 1)
    preds = model.predict(arr)
    return preds

# ------------------ Build combined series: historical + predicted --------------------
hist_years = df['Year'].tolist()
hist_prices = df['Gold_Price_INR'].tolist()
# Determine full range for plotting: from first historical to user-specified horizon
plot_years = list(range(min_year, horizon+1))
pred_prices = []
for y in plot_years:
    if y in hist_years:
        pred_prices.append(float(df.loc[df['Year']==y, 'Gold_Price_INR'].values[0]))
    else:
        # predict with model
        pred_prices.append(float(predict_years([y])[0]))

# ------------------ Layout: Left column (controls + prediction) --------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🔢 Single Year Prediction")
    st.write(f"Predict gold price in India for any year (data trained on {min_year}–{int(df['Year'].max())}).")
    st.write("Enter a year and click Predict.")
    input_year = st.number_input("Year to predict", min_value=min_year, max_value=2100, value=target_year)
    if st.button("Predict Price"):
        pred_val = float(predict_years([input_year])[0])
        # Classification: define tertiles based on historical distribution
        q1, q2 = np.percentile(df['Gold_Price_INR'].values, [33, 66])
        if pred_val <= q1:
            label = "Low"
            color = "green"
        elif pred_val <= q2:
            label = "Medium"
            color = "orange"
        else:
            label = "High"
            color = "red"
        st.markdown(f"### 🎯 Predicted Gold Price for **{input_year}**: ₹{pred_val:,.2f} per 10g")
        st.markdown(f"**Category:** {label}")

    st.markdown("---")
    # Compare Years
    if compare_mode:
        st.subheader("🔎 Compare Two Years")
        year1 = st.number_input("Year 1", min_value=min_year, max_value=2100, value=int(df['Year'].max())-1, key="c1")
        year2 = st.number_input("Year 2", min_value=min_year, max_value=2100, value=int(df['Year'].max()), key="c2")
        if st.button("Compare"):
            price1 = float(predict_years([year1])[0])
            price2 = float(predict_years([year2])[0])
            diff = price2 - price1
            pct = (diff/price1)*100 if price1 != 0 else np.nan
            st.write(f"Price in {year1}: ₹{price1:,.2f} per 10g")
            st.write(f"Price in {year2}: ₹{price2:,.2f} per 10g")
            st.write(f"Difference: ₹{diff:,.2f} ({pct:.2f}%)")

    st.markdown("---")
    # Model info
    st.subheader("📊 Model Info")
    # Compute R2 on historical data
    X_hist = df[['Year']].values
    y_hist = df['Gold_Price_INR'].values
    try:
        y_pred_hist = model.predict(X_hist)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_hist, y_pred_hist)
        mse = mean_squared_error(y_hist, y_pred_hist)
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Training R²: {r2:.4f}")
        st.write(f"Training RMSE: {np.sqrt(mse):.2f}")
    except Exception:
        st.write("Model info: unable to compute metrics.")

# ------------------ Right column: Graph + dataset table --------------------
with right_col:
    st.subheader("📈 Historical Data & Model Predictions")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot historical points
    ax.plot(df['Year'], df['Gold_Price_INR'], marker='o', linestyle='-', label='Historical (actual)', color='#1f77b4')
    # Plot model predictions (entire plot_years)
    ax.plot(plot_years, pred_prices, marker='', linestyle='--', label='Model prediction', color='#ff7f0e')
    # Highlight the user-selected input year if within range
    if input_year in plot_years:
        ysel = float(predict_years([input_year])[0]) if input_year not in hist_years else float(df.loc[df['Year']==input_year, 'Gold_Price_INR'].values[0])
        ax.scatter([input_year], [ysel], color='green', s=100, zorder=5, label=f'Prediction for {input_year}')
    ax.set_xlabel("Year")
    ax.set_ylabel("Gold Price (INR per 10g)")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig)

    # Show dataset table if user wants
    if show_table:
        st.subheader("📚 Dataset (Year vs Price)")
        st.dataframe(df.style.format({"Gold_Price_INR":"{:,}"}), height=300)

# ------------------ PDF Report generation --------------------
def create_pdf_report(year_requested):
    # Generate chart image PNG in-memory
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Prepare textual contents
    pred_for_year = float(predict_years([year_requested])[0])
    q1, q2 = np.percentile(df['Gold_Price_INR'].values, [33, 66])
    if pred_for_year <= q1:
        label = "Low"
    elif pred_for_year <= q2:
        label = "Medium"
    else:
        label = "High"

    # Build PDF using fpdf
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Gold Price Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Requested Year: {year_requested}", ln=True)
    pdf.cell(0, 8, f"Predicted Price (INR per 10g): ₹{pred_for_year:,.2f}", ln=True)
    pdf.cell(0, 8, f"Category: {label}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 8, "Model Info:", ln=True)
    pdf.set_font("Arial", size=10)
    try:
        pdf.cell(0, 6, f"Training R2: {r2:.4f}", ln=True)
        pdf.cell(0, 6, f"Training RMSE: {np.sqrt(mse):.2f}", ln=True)
    except Exception:
        pdf.cell(0, 6, "Training metrics: N/A", ln=True)
    pdf.ln(5)

    # Save chart image to a temp file (fpdf needs a path)
    img_path = "/tmp/gold_plot.png"
    with open(img_path, "wb") as f:
        f.write(buf.getbuffer())

    # Insert the image
    pdf.image(img_path, x=10, y=None, w=190)
    # Output PDF to bytes
    pdf_buf = pdf.output(dest="S").encode("latin-1")
    return pdf_buf

if download_pdf:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Download PDF Report")
    pdf_year = st.sidebar.number_input("PDF report year", min_value=min_year, max_value=2100, value=int(current_year), step=1)
    if st.sidebar.button("Generate & Download PDF"):
        pdf_bytes = create_pdf_report(int(pdf_year))
        st.sidebar.download_button("Download Report (PDF)", data=pdf_bytes, file_name=f"gold_report_{pdf_year}.pdf", mime="application/pdf")

# ------------------ Footer / tips --------------------
st.markdown("---")
st.info("Tips: Use the slider to extend future plotting range. The model is trained on historical data and extrapolates trends; long-term predictions are indicative, not guaranteed.")
