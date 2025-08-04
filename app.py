import streamlit as st
import pandas as pd
import joblib

#Load model and scaler
model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Earthquake Alert Predictor", layout="centered")
st.title("Earthquake Alert Prediction System")
st.markdown("Enter seismic details below to predict the **ALERT LEVEL**.")

# Input fields
latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
depth = st.number_input("Depth (km)", 0.0, 700.0, 10.0)
mag = st.number_input("Magnitude", 4.5, 10.0, 5.0)
magType = st.selectbox("Magnitude Type", ["mb", "ml", "ms", "mw", "mwr", "mwc"])
nst = st.number_input("Number of Stations (nst)", 0, 500, 50)
gap = st.number_input("Azimuthal Gap", 0.0, 360.0, 50.0)
dmin = st.number_input("Minimum Station Distance", 0.0, 20.0, 1.0)
rms = st.number_input("RMS", 0.0, 5.0, 1.0)
horizontalError = st.number_input("Horizontal Error", 0.0, 50.0, 1.0)
depthError = st.number_input("Depth Error", 0.0, 50.0, 1.0)
magError = st.number_input("Magnitude Error", 0.0, 10.0, 0.1)
magNst = st.number_input("Magnitude Station Count", 0, 500, 10)
status = st.selectbox("Status", ["reviewed", "automatic"])
locationSource = st.selectbox("Location Source", ["ci", "us", "hv", "nc", "nm", "se"])
magSource = st.selectbox("Magnitude Source", ["ci", "us", "hv", "nc", "nm", "se"])
type_ = st.selectbox("Type", ["earthquake"])
year = st.number_input("Year", 1976, 2025, 2023)
month = st.slider("Month", 1, 12, 6)
hour = st.slider("Hour", 0, 23, 12)

# Categorical encodings (must match training encoders)
def encode_categorical(col, val):
    enc = {
        'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
        'automatic': 0, 'reviewed': 1,
        'ci': 0, 'hv': 1, 'nc': 2, 'nm': 3, 'se': 4, 'us': 5,
        'earthquake': 0
    }
    return enc.get(val, 0)

# Predict
if st.button("Predict Alert Level"):
    input_data = [[
        latitude, longitude, depth, mag,
        encode_categorical("magType", magType),
        nst, gap, dmin, rms,
        horizontalError, depthError, magError, magNst,
        encode_categorical("status", status),
        encode_categorical("locationSource", locationSource),
        encode_categorical("magSource", magSource),
        encode_categorical("type", type_),
        year, month, hour
    ]]
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    
    # Alert mapping
    alert_map = {0: "green", 1: "orange", 2: "red", 3: "yellow"}
    pred_label = alert_map.get(pred, str(pred)).upper()
    st.success(f"**Predicted Alert Level: {pred_label}**")

    # Explanation of alert levels
    st.markdown("""
    ---  
    **Alert Level Definitions**  
    -**Green**: Minor earthquake with little to no expected impact.  
    -**Yellow**: Moderate potential impact; stay alert.  
    -**Orange**: High potential for damage; caution advised.  
    -**Red**: Severe earthquake with serious threat; take immediate action.
    """)
