import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("earthquake_alert_model.joblib")
scaler = joblib.load("scaler.joblib")

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

# Sidebar
st.sidebar.image("https://img.icons8.com/ios-filled/50/ffffff/earthquakes.png", width=60)
st.sidebar.markdown("## 🌍 Earthquake App")
section = st.sidebar.radio("Choose Page", [
    "📂 Upload & Analyze", "🚨 Single Prediction", "📘 Alert Guide", "⚙️ Settings"
])

# Theme
if "theme" not in st.session_state:
    st.session_state.theme = "light"
theme = st.session_state.theme

# Encode helper
def encode(val):
    enc = {
        'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'mwc': 4, 'mwr': 5,
        'automatic': 0, 'reviewed': 1,
        'ci': 0, 'hv': 1, 'nc': 2, 'nm': 3, 'se': 4, 'us': 5,
        'earthquake': 0
    }
    return enc.get(val, 0)

# Required columns
required_cols = [
    "latitude", "longitude", "depth", "mag", "magType",
    "nst", "gap", "dmin", "rms", "horizontalError",
    "depthError", "magError", "magNst", "status",
    "locationSource", "magSource", "type",
    "year", "month", "hour"
]

# Upload & Analyze
if section == "📂 Upload & Analyze":
    st.title("📂 Upload Earthquake Data and Predict Alerts")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.markdown("### 🧾 File Preview")
        st.dataframe(df.head())

        if all(col in df.columns for col in required_cols):
            # Encode categorical
            df["magType"] = df["magType"].apply(encode)
            df["status"] = df["status"].apply(encode)
            df["locationSource"] = df["locationSource"].apply(encode)
            df["magSource"] = df["magSource"].apply(encode)
            df["type"] = df["type"].apply(encode)

            df_model = df[required_cols].copy()
            df_model.columns = scaler.feature_names_in_
            scaled = scaler.transform(df_model)
            preds = model.predict(scaled)
            alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
            df["Predicted Alert"] = [alert_map.get(p, "UNKNOWN") for p in preds]

            st.success("✅ Predictions completed!")
            st.dataframe(df[["latitude", "longitude", "mag", "depth", "Predicted Alert"]].head())

            st.markdown("### 📊 Alert Distribution")
            count_df = df["Predicted Alert"].value_counts().rename_axis("Alert").reset_index(name="Count")
            fig, ax = plt.subplots()
            ax.bar(count_df["Alert"], count_df["Count"], color=["green", "orange", "red", "gold"])
            ax.set_ylabel("Count")
            ax.set_title("Predicted Alert Level Distribution")
            st.pyplot(fig)

            st.markdown("### 🗺️ Map of Alerts")
            st.map(df[["latitude", "longitude"]])

            # Download options
            st.markdown("### 💾 Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            xlsx = df.to_excel("predictions.xlsx", index=False)
            st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
            st.download_button("📥 Download Excel", open("predictions.xlsx", "rb").read(), "predictions.xlsx")
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.warning(f"""
            ⚠️ Missing required columns:  
            `{', '.join(missing_cols)}`
            """)

# Single Prediction
elif section == "🚨 Single Prediction":
    st.title("🚨 Predict a Single Earthquake Alert")
    with st.form("single_form"):
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

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = [[
            latitude, longitude, depth, mag, encode(magType), nst, gap, dmin,
            rms, 1.0, 1.0, magError, magNst, encode(status),
            0, 0, 0, year, month, hour
        ]]
        df_input = pd.DataFrame(input_data, columns=required_cols)
        df_input.columns = scaler.feature_names_in_
        scaled = scaler.transform(df_input)
        pred = model.predict(scaled)[0]
        alert_map = {0: "GREEN", 1: "ORANGE", 2: "RED", 3: "YELLOW"}
        alert = alert_map.get(pred)
        st.success(f"✅ Predicted Alert: **{alert}**")

        result = df_input.copy()
        result["Predicted Alert"] = alert
        st.dataframe(result)

        # Export
        csv = result.to_csv(index=False).encode('utf-8')
        xlsx_file = "single_prediction.xlsx"
        result.to_excel(xlsx_file, index=False)
        st.download_button("📥 Download Prediction (CSV)", csv, "single_prediction.csv", "text/csv")
        st.download_button("📥 Download Prediction (Excel)", open(xlsx_file, "rb").read(), xlsx_file)

# Guide
elif section == "📘 Alert Guide":
    st.title("📘 Earthquake Alert Level Guide")
    st.markdown("""
    - 🟢 **Green**: Minimal risk  
    - 🟡 **Yellow**: Moderate impact possible  
    - 🟠 **Orange**: High risk — preparation advised  
    - 🔴 **Red**: Severe — immediate action needed  
    """)

# Settings
elif section == "⚙️ Settings":
    st.title("⚙️ App Settings")
    theme_choice = st.radio("Choose Theme", ["light", "dark"], index=0 if theme == "light" else 1)
    st.session_state.theme = theme_choice
    st.markdown(f"<h4 style='color:yellow;'>✅ Theme set to: <b>{theme_choice.upper()} MODE</b></h4>", unsafe_allow_html=True)
