import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

# Set page config
st.set_page_config(page_title="Earthquake Dashboard", layout="wide")

# Sidebar navigation menu
st.sidebar.title("☄️ Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Alert Level", "Alert Definitions", "Settings"])

# Custom styling
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .block-container { padding: 2rem 1rem 2rem 1rem; }
    .header { font-size: 32px; font-weight: bold; color: #1f1f1f; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Navigation logic
if page == "Dashboard":
    st.markdown('<div class="header">📊 Earthquake Dashboard Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🌐 Events Tracked", "9,512", "+12")
    col2.metric("🟢 Low Risk", "6,211", "-5%")
    col3.metric("🔴 High Risk", "1,103", "+14%")
    
    st.markdown("✅ More charts coming soon!")

elif page == "Predict Alert Level":
    st.markdown('<div class="header">🚨 Earthquake Alert Prediction</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            latitude = st.number_input("Latitude", -90.0, 90.0, 35.0)
            longitude = st.number_input("Longitude", -180.0, 180.0, 139.0)
            depth = st.number_input("Depth (km)", 0.0, 700.0, 10.0)
            mag = st.number_input("Magnitude", 4.5, 10.0, 7.5)
            magType = st.selectbox("Magnitude Type", ["mb", "ml", "ms", "mw", "mwr", "mwc"])
            status = st.selectbox("Status", ["reviewed", "automatic"])
        with c2:
            nst = st.number_input("Station Count", 0, 500, 100)
            gap = st.number_input("Azimuthal Gap", 0.0, 360.0, 40.0)
            dmin = st.number_input("Min Station Distance", 0.0, 20.0, 0.5)
            rms = st.number_input("RMS", 0.0, 5.0, 1.1)
            magError = st.number_input("Magnitude Error", 0.0, 10.0, 0.1)
            magNst = st.number_input("MagNst", 0, 500, 20)
        
        year = st.number_input("Year", 1976, 2025, 2023)
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour", 0, 23, 12)

        submitted = st.form_submit_button("Predict")

        if submitted:
            def encode(val, ref):
                enc = {
                    'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
                    'automatic': 0, 'reviewed': 1,
                    'earthquake': 0
                }
                return enc.get(val, 0)

            input_data = [[
                latitude, longitude, depth, mag,
                encode(magType, "magType"),
                nst, gap, dmin, rms, 1.0, 1.0, magError, magNst,
                encode(status, "status"), 0, 0, 0,
                year, month, hour
            ]]
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
            st.success(f"Predicted Alert Level: **{alert_map.get(pred, pred)}**")

elif page == "Alert Definitions":
    st.markdown('<div class="header">📘 Alert Level Definitions</div>', unsafe_allow_html=True)
    st.markdown("""
    - 🟢 **Green**: Minimal risk — informational only.  
    - 🟡 **Yellow**: Moderate impact possible — stay aware.  
    - 🟠 **Orange**: High risk — prepare emergency plans.  
    - 🔴 **Red**: Severe impact expected — take immediate action.
    """)

elif page == "Settings":
    st.markdown('<div class="header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown("Coming soon: model versioning, user preferences, and theme switcher.")


