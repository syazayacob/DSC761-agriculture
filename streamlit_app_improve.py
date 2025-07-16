import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import shap
import plotly.graph_objects as go
import plotly.express as px # New import for interactive time series plots

# For rendering SHAP force plots (which are HTML)
import streamlit.components.v1 as components

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="🌾 Crop Prediction App", layout="centered")
st.title("🌱 Agricultural Prediction System")

st.markdown(
    "Use this app to predict crop **Production**, **Yield**, or **Area harvested** "
    "using different machine learning models."
)

# Define unit map globally for consistent use across predictions and time series
unit_map = {"Production": "tonnes", "Yield": "hg/ha", "Area harvested": "ha"}

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    """
    Loads the crop data from a CSV file.
    Uses st.cache_data to cache the data for faster re-runs.
    """
    try:
        # Ensure the path is correct relative to where the script is run
        # Assuming 'data/improve/crop_data_pivot_log.csv' exists
        df = pd.read_csv("data/improve/crop_data_pivot_log.csv")
        # Ensure 'Year' column is integer for consistent filtering
        df['Year'] = df['Year'].astype(int)
        return df
    except FileNotFoundError:
        st.error("Error: 'crop_data_pivot_log.csv' not found. Please ensure it's in the 'data/improve/' directory.")
        st.stop() # Stop the app if data is not found
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data()
all_elements = ["Production", "Area harvested", "Yield"]

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Prediction Settings")
element_choice = st.sidebar.selectbox("🎯 Target Element", all_elements)
model_choice = st.sidebar.selectbox("🧠 Model", ["ANN", "Random Forest", "Linear Regression", "XGBoost"])

# Define feature columns based on the selected target element
# The target element is excluded from features, and 'Year' is always included.
feature_cols = [col for col in all_elements if col != element_choice] + ["Year"]
feature_cols_str = [str(c) for c in feature_cols] # Ensure column names are strings for consistency

# ----------------------------
# Input Form
# ----------------------------
st.subheader("📥 Input Features")
with st.form("prediction_form"):
    user_inputs = {}
    for col in feature_cols:
        default = 2020 if col == "Year" else 10000.0 # Use float default for consistency
        suffix = " (year)" if col == "Year" else f" ({unit_map.get(col, 'units')})" # Dynamic suffix based on unit_map
        user_inputs[col] = st.number_input(f"{col}{suffix}", value=float(default))

    submitted = st.form_submit_button("🔍 Predict")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    st.write("✅ Input Data:")
    input_df = pd.DataFrame([user_inputs])
    # Ensure column names are strings, which is crucial for consistency with loaded models/scalers
    input_df.columns = input_df.columns.astype(str)
    st.dataframe(input_df)

    with st.spinner("⏳ Predicting..."):
        try:
            base_path = "models" # Assuming 'models' directory exists at the same level as the script
            model_key = f"{element_choice}_{model_choice.replace(' ', '')}"
            scaler_path = os.path.join(base_path, f"{model_key}_scaler.pkl")

            # Load model and scaler based on model choice
            if model_choice == "ANN":
                model_path = os.path.join(base_path, f"{model_key}.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"ANN Model not found at: {model_path}")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler for ANN not found at: {scaler_path}")
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
            else:
                model_bundle_path = os.path.join(base_path, f"{model_key}.pkl")
                if not os.path.exists(model_bundle_path):
                    raise FileNotFoundError(f"Model bundle not found at: {model_bundle_path}")
                bundle = joblib.load(model_bundle_path)
                model, scaler = bundle["model"], bundle["scaler"]

            # Transform input features using the loaded scaler
            # Explicitly cast input_df columns to float before scaling
            input_df_for_scaling = input_df.copy()
            for col in input_df_for_scaling.columns:
                input_df_for_scaling[col] = input_df_for_scaling[col].astype(float)
            X_input = scaler.transform(input_df_for_scaling)
            
            # Make prediction
            # For ANN, prediction result might be a 2D array, flatten it.
            prediction = float(model.predict(X_input).flatten()[0] if model_choice == "ANN" else model.predict(X_input)[0])

            # Format the prediction with appropriate units
            formatted_prediction = f"{prediction:,.0f} {unit_map.get(element_choice, '')}"

            # ----------------------------
            # Prediction Display
            # ----------------------------
            st.subheader("📊 Prediction Result")
            st.success(f"✅ Predicted **{element_choice}**: {formatted_prediction}")

            # ----------------------------
            # Explainable AI (SHAP Visualization)
            # ----------------------------
            st.subheader("🧠 Model Explanation (XAI)")

            if model_choice in ["Random Forest", "XGBoost", "Linear Regression"]:
                try:
                    st.markdown("Understanding how each feature influences the prediction:")

                    # Prepare background data for SHAP explainer
                    # Use a sample of the original data, transformed by the scaler
                    # Ensure background data has the same columns as the input
                    background_df = df[feature_cols].copy()
                    # Explicitly cast background_df columns to float before scaling
                    for col in background_df.columns:
                        background_df[col] = background_df[col].astype(float)

                    # Sample up to 100 rows or the entire dataset if smaller
                    background = pd.DataFrame(
                        scaler.transform(background_df.sample(n=min(100, len(background_df)), random_state=42)),
                        columns=feature_cols_str
                    )

                    # Transform the single input instance for SHAP
                    # Use the already float-casted input_df_for_scaling
                    X_input_transformed = pd.DataFrame(
                        scaler.transform(input_df_for_scaling),
                        columns=feature_cols_str
                    )

                    # Initialize SHAP Explainer based on model type
                    if model_choice in ["Random Forest", "XGBoost"]:
                        explainer = shap.TreeExplainer(model, background)
                    elif model_choice == "Linear Regression":
                        explainer = shap.LinearExplainer(model, background)
                    else:
                        # Fallback for other model types if needed, though less efficient
                        explainer = shap.KernelExplainer(model.predict, background)

                    # Calculate SHAP values for the input instance
                    shap_values = explainer.shap_values(X_input_transformed)

                    # If shap_values is a list (e.g., for multi-output models, or some tree explainers),
                    # take the first element for single-output regression
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # Get the expected value (base value) for the SHAP plots
                    expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else float(np.mean(model.predict(background)))

                    # ------------------------------------------------------------------
                    # SHAP Force Plot (Individual Prediction Explanation)
                    # ------------------------------------------------------------------
                    st.markdown("### 📈 Individual Prediction Explanation (SHAP Force Plot)")
                    st.markdown(
                        "The force plot visualizes how each feature pushes the prediction "
                        "from the base value (average prediction) to the final predicted value."
                    )
                    shap_html = shap.force_plot(
                        expected_value,
                        shap_values[0], # SHAP values for the first (and only) input instance
                        X_input_transformed.iloc[0], # Original feature values for the instance (as Series)
                        feature_names=feature_cols_str,
                        matplotlib=False # Ensure HTML output
                    )
                    # Render the HTML output in Streamlit
                    components.html(shap_html.html(), height=300, scrolling=True)

                    # ------------------------------------------------------------------
                    # SHAP Waterfall Plot (Feature Contribution Breakdown)
                    # ------------------------------------------------------------------
                    st.markdown("### 📊 Feature Contribution Waterfall Plot")
                    st.markdown(
                        "The waterfall plot shows the contribution of each feature "
                        "to the prediction, ordered by magnitude."
                    )
                    # Create a SHAP Explanation object for the waterfall plot
                    explanation_obj = shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=X_input_transformed.iloc[0].values, # .values converts Series to numpy array
                        feature_names=feature_cols_str
                    )
                    fig_waterfall = shap.plots.waterfall(
                        explanation_obj,
                        show=False # Don't show immediately, return figure
                    )
                    st.pyplot(fig_waterfall) # Display Matplotlib figure in Streamlit

                except Exception as ex:
                    st.warning(f"⚠️ SHAP explanation visualization not available for this model/data combination: {str(ex)}")
                    st.info("Ensure the selected model type is compatible with SHAP explainers and data is correctly formatted. Some models or data structures might cause issues.")

            elif model_choice == "ANN":
                st.info("ℹ️ SHAP visualization for ANN models is not directly supported in this app due to computational complexity. Consider using simpler models like Random Forest or XGBoost for better interpretability.")

            # ----------------------------
            # Model Comparison (Conceptual)
            # ----------------------------
            st.subheader("🔄 Model Comparison Notes")
            st.info(
                "To compare models, select a different model from the sidebar and click 'Predict' again "
                "with the same input features. Observe how the predicted value changes. "
                "For a deeper comparison, you would typically evaluate models on a test dataset using metrics like R-squared, Mean Absolute Error (MAE), or Root Mean Squared Error (RMSE)."
            )

            # ----------------------------
            # 3D Visualization
            # ----------------------------
            st.subheader("📉 3D Feature Visualization")
            st.markdown(
                "Visualize the relationship between two input features and the predicted output in 3D."
            )

            # Ensure selected features are in the input_df for plotting
            available_features_for_3d = [f for f in feature_cols if f in input_df.columns]

            if len(available_features_for_3d) >= 2:
                # Allow user to select X and Y axes for the 3D plot
                x_feature = st.selectbox("X-axis", available_features_for_3d, index=0)
                # Ensure Y-axis selection is different from X-axis
                y_options = [f for f in available_features_for_3d if f != x_feature]
                y_feature = st.selectbox("Y-axis", y_options, index=0 if y_options else 0)

                # Create the 3D scatter plot using Plotly Graph Objects
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=input_df[x_feature],
                    y=input_df[y_feature],
                    z=[prediction], # The predicted value on the Z-axis
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=0.8, symbol='circle')
                )])
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=f"{x_feature} ({unit_map.get(x_feature, 'units')})",
                        yaxis_title=f"{y_feature} ({unit_map.get(y_feature, 'units')})",
                        zaxis_title=f"{element_choice} ({unit_map.get(element_choice, 'units')})"
                    ),
                    margin=dict(l=0, r=0, b=0, t=40), # Adjust top margin for title
                    title=f"3D Visualization: {element_choice} Prediction vs. {x_feature} and {y_feature}"
                )
                st.plotly_chart(fig_3d)
            else:
                st.info("Not enough input features available to create a 3D visualization. Need at least two input features.")

        except FileNotFoundError as fnf_e:
            st.error(f"🚫 File not found error: {str(fnf_e)}. Please ensure model and scaler files are in the 'models' directory and data file in 'data/improve'.")
        except Exception as e:
            st.error(f"🚫 An unexpected error occurred during prediction: {str(e)}. Please check your inputs and model files. Details: {type(e).__name__}")

# ----------------------------
# Historical Data & Trends (Time Series)
# ----------------------------
st.subheader("📈 Historical Data & Trends")
st.markdown("Explore historical data for different crop elements, items, and areas to understand past trends.")

# Get unique values for filters from the loaded DataFrame
unique_areas = df["Area"].unique()
unique_items = df["Item"].unique()
unique_elements_ts = df["Element"].unique() # All elements are relevant for historical view

# Time series filters using Streamlit selectboxes
selected_area = st.selectbox(
    "Select Area",
    unique_areas,
    index=np.where(unique_areas == "Afghanistan")[0][0] if "Afghanistan" in unique_areas else 0
)
selected_item = st.selectbox(
    "Select Item",
    unique_items,
    index=np.where(unique_items == "Maize")[0][0] if "Maize" in unique_items else 0
)
selected_element_ts = st.selectbox(
    "Select Element for Trend",
    unique_elements_ts,
    index=np.where(unique_elements_ts == "Production")[0][0] if "Production" in unique_elements_ts else 0
)

# Filter data for the time series plot based on user selections
mask_ts = (df["Area"] == selected_area) & \
          (df["Item"] == selected_item) & \
          (df["Element"] == selected_element_ts)
df_plot_ts = df[mask_ts].sort_values("Year")

if not df_plot_ts.empty:
    # Create an interactive line plot using Plotly Express
    fig_ts = px.line(
        df_plot_ts,
        x="Year",
        y="Value",
        title=f"Historical {selected_item} {selected_element_ts} in {selected_area}",
        labels={"Value": f"{selected_element_ts} ({unit_map.get(selected_element_ts, 'units')})"},
        template="plotly_white" # A clean, professional template
    )
    fig_ts.update_traces(mode='lines+markers', hovertemplate='Year: %{x}<br>Value: %{y:,.0f}') # Add markers and custom hover info
    fig_ts.update_layout(hovermode="x unified") # Unified hover for better UX
    st.plotly_chart(fig_ts, use_container_width=True) # Make plot responsive
else:
    st.info("No historical data found for the selected combination. Please adjust filters.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, Scikit-learn, Keras, SHAP, and Plotly.")
