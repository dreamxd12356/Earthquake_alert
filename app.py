import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Modern Earthquake Dashboard", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #111;
    color: white;
}
h1, h2, h3 {
    color: #003366;
    font-family: 'Segoe UI', sans-serif;
}
.stButton > button {
    background-color: #0052cc;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/ios-filled/50/ffffff/earthquakes.png", width=60)
st.sidebar.markdown("## 🌍 Earthquake App")
section = st.sidebar.radio("Choose Page", [
    "📂 Upload & Analyze",
    "🚨 Single Prediction",
    "📘 Alert Guide",
    "⚙️ Settings"
])

if "theme" not in st.session_state:
    st.session_state.theme = "light"
theme = st.session_state.theme

if section == "📘 Alert Guide":
    st.title("📘 Earthquake Alert Level Guide")
    st.markdown("### Alert Level Definitions")
    st.markdown("- :green_circle: **Green** — Minimal risk")
    st.markdown("- :yellow_circle: **Yellow** — Moderate impact possible")
    st.markdown("- :orange_circle: **Orange** — High risk, prepare accordingly")
    st.markdown("- :red_circle: **Red** — Severe impact, take immediate action")
