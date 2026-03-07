import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt  # Importing for the temperature chart

# Set page configuration
st.set_page_config(page_title="Tomorrow Temperature Predictor", layout="centered")

# Add header and subheader
st.header("University of Zimbabwe")
st.subheader("Department of Analytics and Informatics")

# Project details
st.write("Project:")
st.write("**A Hybrid CNN-GRU Model for Short-Term Temperature Forecasting**")
st.write("Author: Kenward Marambahwenda")

ARTIFACT_DIR = "artifacts"

# -----------------------------
# Load trained artifacts
# -----------------------------
model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "cnn_gru_temp_model.keras"))
scaler_X = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_y.pkl"))
feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))

with open(os.path.join(ARTIFACT_DIR, "config.json"), "r") as f:
    config = json.load(f)

WINDOW = config["window"]

# -----------------------------
# Feature engineering function
# -----------------------------
def add_engineered_features(df):
    df = df.copy()

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fill required numeric columns safely
    numeric_cols = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
        "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Day of year cyclical features
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    # Month cyclical features
    df["month"] = df["Date"].dt.month
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Engineered weather features
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["HumidityMean"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2.0
    df["PressureMean"] = (df["Pressure9am"] + df["Pressure3pm"]) / 2.0
    df["WindSpeedMean"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2.0
    df["CloudMean"] = (df["Cloud9am"] + df["Cloud3pm"]) / 2.0

    return df

# -----------------------------
# App UI
# -----------------------------
st.title("Hybrid CNN-GRU Temperature Forecast")
st.write("Enter today's weather values")

min_temp = st.number_input("MinTemp", value=15.0)
max_temp = st.number_input("MaxTemp", value=28.0)
humidity_3pm = st.number_input("Humidity3pm", value=45.0)

predict = st.button("Predict Tomorrow")

# -----------------------------
# Prediction logic
# -----------------------------
if predict:
    try:
        history_path = os.path.join(ARTIFACT_DIR, "recent_raw_history.csv")
        history = pd.read_csv(history_path)

        # Make sure required raw columns exist
        required_raw_cols = [
            "Date",
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm",
        ]

        missing_raw = [col for col in required_raw_cols if col not in history.columns]
        if missing_raw:
            st.error(f"Missing required columns in recent_raw_history.csv: {missing_raw}")
            st.stop()

        # Ensure enough rows BEFORE adding today's row
        if len(history) < WINDOW - 1:
            st.error(f"recent_raw_history.csv must have at least {WINDOW - 1} rows.")
            st.stop()

        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")

        # Copy last row as template
        latest = history.tail(1).copy()

        # Replace only the user-entered fields
        latest["MinTemp"] = min_temp
        latest["MaxTemp"] = max_temp
        latest["Humidity3pm"] = humidity_3pm

        # Set the new row date to next day
        latest["Date"] = latest["Date"] + pd.Timedelta(days=1)

        # Combine old history + new user row
        combined = pd.concat([history, latest], ignore_index=True)

        # Recreate engineered features used during training
        combined = add_engineered_features(combined)

        # Check if all expected feature columns now exist
        missing_features = [col for col in feature_cols if col not in combined.columns]
        if missing_features:
            st.error(f"These expected feature columns are still missing: {missing_features}")
            st.write("Expected feature columns:", feature_cols)
            st.write("Available columns:", combined.columns.tolist())
            st.stop()

        # Select only training features
        features = combined[feature_cols]

        # Take the last WINDOW rows
        last_window = features.tail(WINDOW)

        if len(last_window) != WINDOW:
            st.error(f"Could not form a full {WINDOW}-day input window.")
            st.stop()

        # Scale and reshape
        X_input = scaler_X.transform(last_window.values)
        X_input = X_input.reshape(1, WINDOW, len(feature_cols))

        # Predict
        pred_scaled = model.predict(X_input, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)

        st.success(f"Predicted Tomorrow MaxTemp: {pred[0][0]:.2f} °C")

        # Optional: Add a temperature chart
        temps = [min_temp, max_temp, pred[0][0]]
        labels = ["MinTemp Today", "MaxTemp Today", "Predicted MaxTemp Tomorrow"]

        plt.figure()
        plt.plot(labels, temps, marker="o")
        plt.title("Temperature Trend")
        plt.ylabel("Temperature °C")

        st.pyplot(plt)

        # Update rolling history buffer back to 29 rows
        combined.tail(WINDOW - 1)[required_raw_cols].to_csv(history_path, index=False)

        with st.expander("Debug info"):
            st.write("Window size:", WINDOW)
            st.write("Expected feature columns:", feature_cols)
            st.write("Last window shape:", last_window.shape)
            st.write("Latest prediction input row:")
            st.dataframe(latest)

    except Exception as e:
        st.error(f"An error occurred: {e}")