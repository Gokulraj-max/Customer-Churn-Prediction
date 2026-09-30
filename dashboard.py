"""
Customer Churn Prediction - Interactive Streamlit Dashboard
Allows business users and stakeholders to input customer information,
run real-time inference against the trained model or FastAPI service,
and explore churn risk factors.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

# Load local model cache for direct fallback inference
@st.cache_resource
def load_local_pipeline():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

local_pipeline = load_local_pipeline()

# Title and Description
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    """
    Predict the probability that a telecommunications customer will terminate their service (churn).
    This tool utilizes a trained **Machine Learning pipeline** with automated data preprocessing,
    feature encoding, and probability calibration.
    """
)

# Sidebar: Configuration & Presets
st.sidebar.header("⚙️ Settings & Presets")
inference_mode = st.sidebar.radio(
    "Inference Engine",
    options=["FastAPI Service (REST)", "Embedded Local Model"],
    index=0 if local_pipeline is None else 1,
    help="FastAPI calls the live microservice; Local Model runs inference directly in Streamlit."
)
api_endpoint = st.sidebar.text_input("FastAPI Endpoint URL", value=DEFAULT_API_URL)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Quick Test Profiles")
preset = st.sidebar.selectbox(
    "Load Example Preset",
    ["Custom Input", "High Churn Risk (Month-to-month, High Bills)", "Low Churn Risk (2-Year Contract, Retained)"]
)

# Preset data dictionary
preset_values = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0
}

if preset == "Low Churn Risk (2-Year Contract, Retained)":
    preset_values = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3300.0
    }

# Input Form in 3 structured columns
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Customer Demographics")
        gender = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female", "Male"].index(preset_values["gender"])
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            index=[0, 1].index(preset_values["SeniorCitizen"]),
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
        )
        partner = st.selectbox(
            "Has Partner", ["Yes", "No"],
            index=["Yes", "No"].index(preset_values["Partner"])
        )
        dependents = st.selectbox(
            "Has Dependents", ["Yes", "No"],
            index=["Yes", "No"].index(preset_values["Dependents"])
        )
        tenure = st.slider(
            "Tenure (Months with company)", 0, 72,
            value=preset_values["tenure"]
        )

    with col2:
        st.subheader("📶 Subscribed Services")
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"],
            index=["Yes", "No"].index(preset_values["PhoneService"])
        )
        lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"],
            index=["No", "Yes", "No phone service"].index(preset_values["MultipleLines"])
        )
        internet = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"],
            index=["Fiber optic", "DSL", "No"].index(preset_values["InternetService"])
        )
        sec = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["OnlineSecurity"])
        )
        backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["OnlineBackup"])
        )
        device = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["DeviceProtection"])
        )
        support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["TechSupport"])
        )
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["StreamingTV"])
        )
        movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(preset_values["StreamingMovies"])
        )

    with col3:
        st.subheader("💳 Account & Billing")
        contract = st.selectbox(
            "Contract Term", ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(preset_values["Contract"])
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"],
            index=["Yes", "No"].index(preset_values["PaperlessBilling"])
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ].index(preset_values["PaymentMethod"])
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=300.0,
            value=float(preset_values["MonthlyCharges"]),
            step=1.0
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=15000.0,
            value=float(preset_values["TotalCharges"]),
            step=10.0
        )

    submitted = st.form_submit_button("🚀 Analyze Churn Risk", use_container_width=True)

# Handle Form Submission
if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": sec,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    prediction = None
    probability = None
    risk_level = None
    message = None

    if "FastAPI" in inference_mode:
        try:
            res = requests.post(api_endpoint, json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                prediction = data["churn_prediction"]
                probability = data["churn_probability"]
                risk_level = data["risk_level"]
                message = data["message"]
            else:
                st.error(f"API Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.warning(f"Could not connect to FastAPI at {api_endpoint}. Falling back to embedded local model. Error: {e}")

    # Fallback to local model if API mode failed or Embedded mode selected
    if probability is None:
        if local_pipeline is not None:
            df_in = pd.DataFrame([payload])
            prediction = int(local_pipeline.predict(df_in)[0])
            probability = float(local_pipeline.predict_proba(df_in)[0][1])
            if probability >= 0.70:
                risk_level, message = "High", "High churn probability. Priority customer retention strategy required."
            elif probability >= 0.40:
                risk_level, message = "Medium", "Moderate risk. Offer targeted retention perks and incentives."
            else:
                risk_level, message = "Low", "Customer is likely to stay. Standard relationship engagement."
        else:
            st.error("No model available! Please run 'python src/train.py' first.")

    # Render Results
    if probability is not None:
        st.markdown("---")
        st.header("🎯 Churn Risk Assessment")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label="Predicted Outcome", value="Likely Churn (1)" if prediction == 1 else "Retained (0)")
        with res_col2:
            st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")
        with res_col3:
            st.metric(label="Risk Category", value=risk_level)

        # Risk indicator banner
        if risk_level == "High":
            st.error(f"⚠️ **HIGH RISK**: {message}")
        elif risk_level == "Medium":
            st.warning(f"⚡ **MEDIUM RISK**: {message}")
        else:
            st.success(f"✅ **LOW RISK**: {message}")

        # Visual progress bar
        st.progress(probability)
