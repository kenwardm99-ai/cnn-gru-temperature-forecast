import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Today Temperature Predictor",
    layout="centered"
)

ARTIFACT_DIR = "artifacts"

# ---------------------------------------------------
# Load trained artifacts safely
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(ARTIFACT_DIR, "cnn_gru_temp_model.keras"),
        compile=False
    )
    scaler_X = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_y.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))

    with open(os.path.join(ARTIFACT_DIR, "config.json"), "r") as f:
        config = json.load(f)

    return model, scaler_X, scaler_y, feature_cols, config


model, scaler_X, scaler_y, feature_cols, config = load_artifacts()
WINDOW = config["window"]  # this should now be 7 in config.json

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.header("University of Zimbabwe")
st.subheader("Department of Analytics and Informatics")
st.write("**Project:** A Hybrid CNN-GRU Model for Short-Term Temperature Forecasting")
st.write("**Author:** Kenward Marambahwenda")

# ---------------------------------------------------
# Sidebar: model information
# ---------------------------------------------------
st.sidebar.title("Model Information")
st.sidebar.write("**Model Type:** Hybrid CNN-GRU")
st.sidebar.write(f"**Input Window:** {WINDOW} days")
st.sidebar.write("**Prediction Target:** Today's temperature using yesterday's weather values")
st.sidebar.write("**Features Used:** 25 engineered meteorological features")

with st.sidebar.expander("View full feature list"):
    st.write(", ".join(feature_cols))

# ---------------------------------------------------
# Feature engineering function
# ---------------------------------------------------
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

# ---------------------------------------------------
# Main app title and description
# ---------------------------------------------------
st.title("Hybrid CNN-GRU Temperature Forecast")
st.write("Enter yesterday's weather conditions to predict today's temperature.")

# ---------------------------------------------------
# Dashboard-style input layout
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    min_temp = st.number_input("Yesterday MinTemp (°C)", value=15.0)

with col2:
    max_temp = st.number_input("Yesterday MaxTemp (°C)", value=28.0)

with col3:
    humidity_3pm = st.number_input("Yesterday Humidity3pm (%)", value=45.0)

predict = st.button("Predict Today")

# ---------------------------------------------------
# Prediction logic
# ---------------------------------------------------
if predict:
    try:
        history_path = os.path.join(ARTIFACT_DIR, "recent_raw_history.csv")
        history = pd.read_csv(history_path)

        # Required raw columns
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

        # Need WINDOW - 1 rows before adding yesterday's row
        if len(history) < WINDOW - 1:
            st.error(f"recent_raw_history.csv must have at least {WINDOW - 1} rows.")
            st.stop()

        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")

        # Use latest row as template
        latest = history.tail(1).copy()

        # Replace only user-entered fields using yesterday's values
        latest["MinTemp"] = min_temp
        latest["MaxTemp"] = max_temp
        latest["Humidity3pm"] = humidity_3pm

        # Advance date by one day so that yesterday's input becomes today's forecast context
        latest["Date"] = latest["Date"] + pd.Timedelta(days=1)

        # Build full 7-day sequence
        combined = pd.concat([history, latest], ignore_index=True)

        # Recreate engineered features
        combined = add_engineered_features(combined)

        # Check expected features
        missing_features = [col for col in feature_cols if col not in combined.columns]
        if missing_features:
            st.error(f"These expected feature columns are still missing: {missing_features}")
            st.write("Expected feature columns:", feature_cols)
            st.write("Available columns:", combined.columns.tolist())
            st.stop()

        # Select training features
        features = combined[feature_cols]

        # Take last WINDOW rows
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

        # Handle single-output or two-output model
        pred_max = pred[0][0]
        pred_min = pred[0][1] if pred.shape[1] > 1 else None

        # ---------------------------------------------------
        # Forecast result section
        # ---------------------------------------------------
        st.subheader("Forecast Result")

        if pred_min is not None:
            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric("Yesterday MinTemp", f"{min_temp:.2f} °C")
            with r2:
                st.metric("Yesterday MaxTemp", f"{max_temp:.2f} °C")
            with r3:
                st.metric("Predicted Today MinTemp", f"{pred_min:.2f} °C")
            with r4:
                st.metric("Predicted Today MaxTemp", f"{pred_max:.2f} °C")
        else:
            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Yesterday MinTemp", f"{min_temp:.2f} °C")
            with r2:
                st.metric("Yesterday MaxTemp", f"{max_temp:.2f} °C")
            with r3:
                st.metric("Predicted Today Temperature", f"{pred_max:.2f} °C")

        # ---------------------------------------------------
        # Temperature chart
        # ---------------------------------------------------
        st.subheader("Temperature Trend")

        if pred_min is not None:
            temps = [min_temp, max_temp, pred_min, pred_max]
            labels = [
                "Yesterday MinTemp",
                "Yesterday MaxTemp",
                "Predicted Today MinTemp",
                "Predicted Today MaxTemp"
            ]
        else:
            temps = [min_temp, max_temp, pred_max]
            labels = [
                "Yesterday MinTemp",
                "Yesterday MaxTemp",
                "Predicted Today Temperature"
            ]

        fig, ax = plt.subplots()
        ax.plot(labels, temps, marker="o")
        ax.set_title("Temperature Trend")
        ax.set_ylabel("Temperature (°C)")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        # Update rolling history buffer back to WINDOW - 1 rows
        combined.tail(WINDOW - 1)[required_raw_cols].to_csv(history_path, index=False)

        # ---------------------------------------------------
        # Debug section
        # ---------------------------------------------------
        with st.expander("Debug info"):
            st.write("Window size:", WINDOW)
            st.write("Expected feature columns:", feature_cols)
            st.write("Last window shape:", last_window.shape)
            st.write("Prediction array:", pred)
            st.write("Latest prediction input row:")
            st.dataframe(latest)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("Developed by Kenward Marambahwenda | University of Zimbabwe | Department of Analytics and Informatics")
