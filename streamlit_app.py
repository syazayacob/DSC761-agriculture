import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Crop Prediction", layout="centered")
st.title("🌾 Agricultural Prediction System")

st.write("Select a target (element) and model to predict crop data such as Production, Yield, or Area harvested.")

# ----------------------------
# User selections
# ----------------------------
element_choice = st.selectbox("Choose Target Element", ["Production", "Yield", "Area harvested"])

model_choice = st.selectbox(
    "Select Model",
    ["ANN", "Random Forest", "Linear Regression", "XGBoost"]
)

# ----------------------------
# Build model path
# ----------------------------
model_key = f"{element_choice.lower().replace(' ', '_')}_{model_choice.lower().replace(' ', '')}"
model_path = f"models/{model_key}.pkl"

# ----------------------------
# Check if model exists
# ----------------------------
if not os.path.exists(model_path):
    st.error(f"🚫 Model not found: `{model_path}`. Please check your models folder.")
    st.stop()

# Load the model
model = joblib.load(model_path)

# ----------------------------
# Input form
# ----------------------------
with st.form("prediction_form", clear_on_submit=False):
    st.subheader("📅 Enter Year")
    year = st.number_input("Year", min_value=1961, max_value=2025, value=2020, step=1)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_df = pd.DataFrame({"Year": [year]})

    st.subheader("🔎 Model Info")
    st.write(f"Model file: `{model_path}`")
    st.write("Input Data:")
    st.dataframe(input_df)

    # Run prediction
    prediction = model.predict(input_df)[0]

    # Define unit map
    unit_map = {
        "Production": "tonnes",
        "Yield": "hg/ha",
        "Area harvested": "ha"
    }

    unit = unit_map.get(element_choice, "")
    formatted_result = f"{element_choice} = {prediction:,.0f} {unit}"

    st.subheader("📊 Prediction Result")
    st.success(f"✅ Prediction: **{formatted_result}**")
