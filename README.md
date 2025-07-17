# DSC761-agriculture

# 🌾 Agricultural Crop Prediction System

This repository hosts a machine learning project focused on predicting key agricultural metrics: **Production**, **Area harvested**, and **Yield**. It includes data processing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit application for interactive predictions and historical data visualization.

## 📁 Folder Structure
```plaintext
DSX761-agriculture/
├── data/
│ ├── crop1.csv # Raw input dataset
│ └── improve/ # Processed data outputs
│ ├── crop1_clean.csv
│ ├── crop_data_pivot_log.csv
│ └── crop_data_pivot.csv
├── models/
│ └── improve/ # Trained model files (.h5 for ANN, .pkl for others)
│ ├── Production_ANN.h5
│ ├── Production_ANN_scaler.pkl
│ ├── Production_RandomForest.pkl
│ └── ... (other models and scalers for Area harvested, Yield)
├── notebooks/ # Source code for data processing, training, etc.
│ ├── 012_data_preprocessing_improve.ipynb
│ ├── 022_visualization.ipynb
│ ├── 032_model_training.ipynb
│ └── 042_evaluation.ipynb
├── streamlit/
│ └── streamlit_app_improve.py # Streamlit dashboard for predictions and trends
├── requirements.txt # Required Python packages
└── README.md
```





---

## 📊 Contents

### 1. Data Processing
The `012_data_preprocessing_improve.ipynb` script performs the following steps:
- **Filtering**: Selects relevant elements ('Production', 'Area harvested', 'Yield') and years (1961-2020).
- **Missing Value Handling**: Drops rows with null values in the 'Value' column.
- **Winsorization**: Applies winsorization to 'Value' to handle outliers.
- **Log Transformation**: Transforms winsorized values using `log(1+x)` for better distribution.
- **Decade Binning**: Creates a 'Decade' column from the 'Year'.
- **Pivoting**: Creates pivoted tables for raw and log-transformed values, ready for modeling.
- **Saving**: Exports cleaned and pivoted datasets to `data/improve/`.

### 2. Visualization
The `022_visualization.ipynb` script generates various plots to understand the data:
- **Value Distribution**: Histograms comparing original, winsorized, and log-transformed values.
- **Element Count by Decade**: Bar plot showing the frequency of different elements across decades.
- **Top 10 Items**: Bar plot of the most frequently reported crop items.
- **Log-Transformed Value Distribution**: Box plots of log-transformed values by element.
- **Correlation Heatmap**: Visualizes correlations between 'Production', 'Area harvested', and 'Yield'.
- **Time Series Plot**: Shows historical trends for specific Area, Item, and Element combinations (e.g., Maize Production in Afghanistan).

### 3. Model Training
The `032_model_training.ipynb` script trains multiple regression models to predict 'Production', 'Area harvested', and 'Yield':
- **Models**:
    - Artificial Neural Network (ANN)
    - Random Forest Regressor
    - Linear Regression
    - XGBoost Regressor
- **Process**: For each target, models are trained using other elements and 'Year' as features. Data is scaled using `StandardScaler` and split into training/testing sets.
- **Output**: Trained models and their respective scalers are saved to `models`.

### 4. Model Evaluation
The `042_evaluation.ipynb` script loads the trained models and evaluates their performance on the test set:
- **Metrics**: Reports Mean Squared Error (MSE) and R² Score for each model and target.
- **Visualizations**: Generates scatter plots of actual vs. predicted values and a bar plot comparing R² scores across all models and targets.

### 5. Streamlit Application
The `streamlit_app_improve.py` provides an interactive web interface:
- **Prediction**: Users can input features and get predictions for 'Production', 'Area harvested', or 'Yield' using different trained models.
- **Explainable AI (XAI)**: Offers simple feature importance visualizations for tree-based and linear models (bar charts).
- **3D Visualization**: Allows users to visualize the relationship between two input features and the predicted output in a 3D scatter plot.
- **Historical Trends**: Displays interactive time series plots of historical crop data based on user-selected Area, Item, and Element.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.10+ installed.

### Setup
1. **Clone the repository:**
   git clone [https://github.com/syazayacob/DSC761-agriculture.git]
   cd DSC761-agriculture

2. **Install dependencies:**
    It's recommended to create a virtual environment first:

    `pip install -r requirements.txt`

Running the Project
1. Data Processing
    First, run the data processing script to prepare the datasets:
    `/notebooks/012_data_preprocessing_improve.ipynb`

    This will create the data/improve directory and save the processed CSVs there.

2. Model Training
    Next, train the machine learning models:

    `/notebooks/032_model_training.ipynb`

    This will create the models/improve directory and save all trained models and scalers.

3. Model Evaluation (Optional, but recommended)
    To see detailed evaluation metrics and plots:

    `/notebooks/042_evaluation.ipynb`

4. Launch the Streamlit App
    Finally, run the interactive prediction dashboard:

    `streamlit run streamlit_app_improve.py`
    This will open the application in the web browser.
--
## ✨ Credits
    This project was developed to demonstrate a complete machine learning pipeline for agricultural data, from raw data processing to interactive deployment with explainable AI.
