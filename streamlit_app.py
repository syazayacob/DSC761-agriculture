import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.express as px
import plotly.graph_objects as go


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
    df = pd.read_csv("data/processed/crop_data_pivot.csv")
    return df

df = load_data()

# All target elements
all_elements = ["Production", "Area harvested", "Yield"]

# ----------------------------
# User Selections
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


    try:
        # ----------------------------
        # Load Model and Scaler
        # ----------------------------
        # Set correct paths
        base_path = "notebooks/models"
        model_key = f"{element_choice}_{model_choice.replace(' ', '')}"
        scaler_path = os.path.join(base_path, f"{model_key}_scaler.pkl")


        if model_choice == "ANN":
            model_path = os.path.join(base_path, f"{model_key}.h5")


            if not os.path.exists(model_path):
                raise FileNotFoundError("❌ ANN model file not found.")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError("❌ ANN scaler file not found.")

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            X_input = scaler.transform(input_df)
            #prediction = model.predict(X_input).flatten()[0]
            prediction = float(model.predict(X_input).flatten()[0])

        else:
            model_path = os.path.join(base_path, f"{model_key}.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found.")

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle["scaler"]

            X_input = scaler.transform(input_df)
            #prediction = model.predict(X_input)[0]
            prediction = float(model.predict(X_input)[0])

        # Units display
        unit_map = {
            "Production": "tonnes",
            "Yield": "hg/ha",
            "Area harvested": "ha"
        }

        unit = unit_map.get(element_choice, "")
        formatted = f"{prediction:,.0f} {unit}"

        # Display Result
        st.subheader("📊 Prediction Result")
        st.success(f"✅ Predicted **{element_choice}**: {formatted}")


        # ----------------------------
        # Explainable AI (SHAP) - XAI
        # ----------------------------
        if model_choice in ["Random Forest", "XGBoost", "Linear Regression"]:
            try:
                st.subheader("🧠 Model Explanation (XAI)")
                # Sample background data
                background = pd.DataFrame(scaler.transform(df[feature_cols].sample(n=50, random_state=42)))
                X_input_transformed = scaler.transform(input_df)

                # Choose appropriate SHAP explainer
                if model_choice in ["Random Forest", "XGBoost"]:
                    explainer = shap.TreeExplainer(model, background)
                elif model_choice == "Linear Regression":
                    explainer = shap.LinearExplainer(model, background)
                else:
                    explainer = shap.Explainer(model, background)

                shap_values = explainer(X_input_transformed)

                # Waterfall plot using explicit fig
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_values[0], max_display=5, show=False)
                st.pyplot(bbox_inches='tight')

            except Exception as ex:
                st.warning(f"⚠️ SHAP explanation not available: {str(ex)}")


        # ----------------------------
        # Multidimensional Visualization - 3D
        # ----------------------------
        st.subheader("📉 3D Feature Visualization")

        # Select 2 features to visualize against the prediction
        x_feature = st.selectbox("X-axis Feature", feature_cols, index=0, key="x_axis")
        y_feature = st.selectbox("Y-axis Feature", feature_cols, index=1 if len(feature_cols) > 1 else 0, key="y_axis")

        fig = go.Figure(data=[go.Scatter3d(
            x=input_df[x_feature],
            y=input_df[y_feature],
            z=[prediction],
            mode='markers',
            marker=dict(size=8, color='red'),
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=element_choice
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        st.plotly_chart(fig)


    except Exception as e:
        st.error(f"🚫 Error during prediction: {str(e)}")