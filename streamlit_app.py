import streamlit as st
import pandas as pd
import joblib
import os

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="🌾 Crop Prediction App", layout="centered")
st.title("🌱 Agricultural Prediction System")

st.markdown(
    "Use this app to predict crop **Production**, **Yield**, or **Area harvested** "
    "using different machine learning models."
)

# ----------------------------
# Load data to get columns
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/data/processed/crop_data_pivot.csv")
    return df

df = load_data()

# All target elements
all_elements = ["Production", "Area harvested", "Yield"]

# ----------------------------
# User selections
# ----------------------------
st.sidebar.header("🔧 Prediction Settings")
element_choice = st.sidebar.selectbox("🎯 Choose Target Element", all_elements)
model_choice = st.sidebar.selectbox(
    "🧠 Select Model",
    ["ANN", "Random Forest", "Linear Regression", "XGBoost"]
)

# ----------------------------
# Determine required features
# ----------------------------
feature_cols = [col for col in all_elements if col != element_choice] + ["Year"]

# ----------------------------
# Input Form
# ----------------------------
st.subheader("📥 Input Features")
with st.form("prediction_form"):
    user_inputs = {}
    for col in feature_cols:
        default_val = 2020 if col == "Year" else 10000
        user_inputs[col] = st.number_input(f"{col}", value=default_val)

    submitted = st.form_submit_button("🔍 Predict")

# ----------------------------
# Run Prediction
# ----------------------------
if submitted:
    input_df = pd.DataFrame([user_inputs])
    st.write("✅ Input Data:")
    st.dataframe(input_df)

    model_key = f"{element_choice}_ANN" if model_choice == "ANN" else f"{element_choice}_{model_choice}"
    model_dir = "models"

    try:
        # ----------------------------
        # Load Model and Scaler
        # ----------------------------
        if model_choice == "ANN":
            model_path = os.path.join(model_dir, f"{model_key}.h5")
            scaler_path = os.path.join(model_dir, f"{model_key}_scaler.pkl")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError("Model or scaler file not found.")

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

        else:
            model_path = os.path.join(model_dir, f"{model_key}.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found.")

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle["scaler"]

        # ----------------------------
        # Predict
        # ----------------------------
        X_input = scaler.transform(input_df)

        if model_choice == "ANN":
            prediction = model.predict(X_input).flatten()[0]
        else:
            prediction = model.predict(X_input)[0]

        # ----------------------------
        # Display Result
        # ----------------------------
        unit_map = {
            "Production": "tonnes",
            "Yield": "hg/ha",
            "Area harvested": "ha"
        }

        unit = unit_map.get(element_choice, "")
        formatted_prediction = f"{prediction:,.0f} {unit}"

        st.subheader("📊 Prediction Result")
        st.success(f"**Predicted {element_choice}: {formatted_prediction}**")

    except Exception as e:
        st.error(f"🚫 Error during prediction: {str(e)}")