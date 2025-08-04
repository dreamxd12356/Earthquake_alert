import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

# Default values for missing columns
default_values = {
    "latitude": 0.0,
    "longitude": 0.0,
    "depth": 10.0,
    "mag": 5.0,
    "magType": 3,
    "nst": 20,
    "gap": 45.0,
    "dmin": 1.0,
    "rms": 1.0,
    "horizontalError": 1.0,
    "depthError": 1.0,
    "magError": 0.2,
    "magNst": 20,
    "status": 1,
    "locationSource": 0,
    "magSource": 0,
    "type": 0,
    "year": 2023,
    "month": 6,
    "hour": 12
}

# Set up page
st.set_page_config(page_title="Modern Earthquake Dashboard", layout="wide")

# CSS Styling
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

if section == "📂 Upload & Analyze":
    st.title("📂 Upload Earthquake CSV and Predict Alerts")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🦾 File Preview")
        st.dataframe(df.head())

        # Encode categoricals if present
        def encode(val):
            enc = {
                'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
                'automatic': 0, 'reviewed': 1,
                'ci': 0, 'hv': 1, 'nc': 2, 'nm': 3, 'se': 4, 'us': 5,
                'earthquake': 0
            }
            return enc.get(val, 0)

        encode_cols = ["magType", "status", "locationSource", "magSource", "type"]
        for col in encode_cols:
            if col in df.columns:
                df[col] = df[col].apply(encode)

        # Handle missing columns with defaults
        expected_cols = scaler.feature_names_in_
        missing_cols = [col for col in expected_cols if col not in df.columns]

        for col in missing_cols:
            df[col] = default_values.get(col, 0)

        df = df.reindex(columns=expected_cols)
        df_scaled = scaler.transform(df)
        preds = model.predict(df_scaled)

        alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
        df["Predicted Alert"] = [alert_map.get(p, "UNKNOWN") for p in preds]

        st.success("✅ Predictions completed!")
        st.dataframe(df[["latitude", "longitude", "mag", "depth", "Predicted Alert"]].head())

        if missing_cols:
            st.warning("""
            Some columns were missing from your file and filled with default values:
            - {}
            """.format(", ".join(missing_cols)))

        # Download button
        st.download_button("Download Predictions CSV", df.to_csv(index=False), "predictions.csv")

        # Map View
        st.markdown("### 🗺️ Earthquake Prediction Map")
        st.map(df[["latitude", "longitude"]])

        # Bar chart
        st.markdown("### 📊 Alert Distribution")
        count_df = df["Predicted Alert"].value_counts().rename_axis("Alert").reset_index(name="Count")
        fig, ax = plt.subplots()
        ax.bar(count_df["Alert"], count_df["Count"], color=["green", "orange", "red", "gold"])
        ax.set_ylabel("Count")
        ax.set_title("Predicted Alert Level Distribution")
        st.pyplot(fig)

elif section == "🚨 Single Prediction":
    st.title("Predict a Single Earthquake Alert Level")
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
            magNst = st.number_input("MagNst", 0, 500, 20)
        year = st.number_input("Year", 1976, 2025, 2023)
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour", 0, 23, 12)
        submitted = st.form_submit_button("Predict")

    if submitted:
        def encode(val):
            enc = {
                'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
                'automatic': 0, 'reviewed': 1,
                'earthquake': 0
            }
            return enc.get(val, 0)

        input_dict = default_values.copy()
        input_dict.update({
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "mag": mag,
            "magType": encode(magType),
            "nst": nst,
            "gap": gap,
            "dmin": dmin,
            "rms": rms,
            "magError": magError,
            "magNst": magNst,
            "status": encode(status),
            "year": year,
            "month": month,
            "hour": hour
        })

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=scaler.feature_names_in_)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
        st.success(f"✅ Predicted Alert Level: **{alert_map.get(pred)}**")

elif section == "📘 Alert Guide":
    st.title("📘 Earthquake Alert Level Guide")
    st.markdown("""
    - 🟢 **Green**: Minimal risk  
    - 🟡 **Yellow**: Moderate impact possible  
    - 🟠 **Orange**: High risk — preparation advised  
    - 🔴 **Red**: Severe — immediate action needed  
    """)

elif section == "⚙️ Settings":
    st.title("⚙️ App Settings")
    theme_choice = st.radio("Choose Theme", ["light", "dark"], index=0 if theme == "light" else 1)
    st.session_state.theme = theme_choice
    st.markdown(f"<h4 style='color:yellow;'>✅ Theme set to: <b>{theme_choice.upper()} MODE</b></h4>", unsafe_allow_html=True)
