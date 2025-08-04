import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

# Page config
st.set_page_config(page_title="Earthquake Alert System", layout="wide")

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Analyze", "Predict Single Alert", "Alert Definitions", "Settings"
])

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
theme = st.session_state.theme

# Apply theme styles
if theme == "dark":
    st.markdown("""
    <style>
    body, .stApp { background-color: #1e1e1e; color: white; }
    .css-1d391kg, .css-18e3th9 { background-color: #1e1e1e; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body, .stApp { background-color: #f5f7fa; color: black; }
    .css-1d391kg, .css-18e3th9 { background-color: white; color: black; }
    </style>
    """, unsafe_allow_html=True)

# Upload and Analyze
if page == "Upload & Analyze":
    st.title("📂 Upload Earthquake CSV and Predict Alerts")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🧾 Preview")
        st.dataframe(df.head())

        required_cols = ["latitude", "longitude", "depth", "mag", "magType",
                         "nst", "gap", "dmin", "rms", "horizontalError",
                         "depthError", "magError", "magNst", "status",
                         "locationSource", "magSource", "type",
                         "year", "month", "hour"]

        if all(col in df.columns for col in required_cols):
            def encode(val):
                enc = {
                    'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
                    'automatic': 0, 'reviewed': 1,
                    'ci': 0, 'hv': 1, 'nc': 2, 'nm': 3, 'se': 4, 'us': 5,
                    'earthquake': 0
                }
                return enc.get(val, 0)

            # Encode categories
            df["magType"] = df["magType"].apply(encode)
            df["status"] = df["status"].apply(encode)
            df["locationSource"] = df["locationSource"].apply(encode)
            df["magSource"] = df["magSource"].apply(encode)
            df["type"] = df["type"].apply(encode)

            # Predict
            features = df[required_cols]
            scaled = scaler.transform(features)
            preds = model.predict(scaled)
            alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
            df["Predicted Alert"] = [alert_map.get(p, "UNKNOWN") for p in preds]

            st.success("✅ Prediction complete!")
            st.dataframe(df[["latitude", "longitude", "mag", "depth", "Predicted Alert"]].head(10))

            st.markdown("### 📊 Alert Distribution")
            summary = df["Predicted Alert"].value_counts().rename_axis("Alert").reset_index(name="Count")
            fig, ax = plt.subplots()
            ax.bar(summary["Alert"], summary["Count"], color=["green", "orange", "red", "gold"])
            ax.set_ylabel("Count")
            ax.set_title("Predicted Alert Level Distribution")
            st.pyplot(fig)
        else:
            st.error("❌ Missing required columns in the uploaded file.")

# Single Prediction
elif page == "Predict Single Alert":
    st.title("🚨 Predict Earthquake Alert Level")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", -90.0, 90.0, 10.0)
            longitude = st.number_input("Longitude", -180.0, 180.0, 70.0)
            depth = st.number_input("Depth (km)", 0.0, 700.0, 10.0)
            mag = st.number_input("Magnitude", 4.5, 10.0, 6.0)
            magType = st.selectbox("Magnitude Type", ["mb", "ml", "ms", "mw", "mwr", "mwc"])
            status = st.selectbox("Status", ["reviewed", "automatic"])
        with col2:
            nst = st.number_input("Station Count", 0, 500, 100)
            gap = st.number_input("Azimuthal Gap", 0.0, 360.0, 45.0)
            dmin = st.number_input("Min Station Distance", 0.0, 20.0, 1.0)
            rms = st.number_input("RMS", 0.0, 5.0, 1.0)
            magError = st.number_input("Magnitude Error", 0.0, 10.0, 0.2)
            magNst = st.number_input("MagNst", 0,_
