import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from feature_engineering import create_features
from xai import explain_xgboost, clean_model_input

load_dotenv()

st.set_page_config(page_title="Dharali Rainfall Prediction", layout="wide")

LAT, LON = 31.040698, 78.7972018
TFT_WEIGHT = 0.85
XGB_WEIGHT = 0.15

@st.cache_resource
def load_models():
    xgb = joblib.load("models/xgboost_model.pkl")
    tft = TemporalFusionTransformer.load_from_checkpoint(
        "models/tft_model.ckpt",
        map_location="cpu"
    )
    params = joblib.load("models/tft_dataset_params.pkl")

    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    return xgb, tft, params, feature_cols

xgb, tft, tft_params, feature_cols = load_models()

@st.cache_data(ttl=3600)
def get_weather():
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "dew_point_2m", "surface_pressure", "precipitation",
            "rain", "cloud_cover", "wind_speed_10m",
            "wind_direction_10m", "wind_gusts_10m",
            "total_column_integrated_water_vapour",
            "boundary_layer_height", "vapour_pressure_deficit",
            "wet_bulb_temperature_2m"
        ]),
        "past_hours": 72,
        "forecast_hours": 24,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def predict_tft(history, params):
    data = history.copy()
    time_col = params["time_idx"]
    target_col = params["target"]
    group_ids = params["group_ids"]

    # Assumes training time_idx is continuous hourly from this origin.
    origin = pd.Timestamp("2020-01-01 00:00:00")
    data[time_col] = (
        (data["time"] - origin).dt.total_seconds() // 3600
    ).astype(int)

    data[target_col] = 0.0

    for group in group_ids:
        data[group] = "Dharali"

    dataset = TimeSeriesDataSet.from_parameters(
        params,
        data,
        predict=True,
        stop_randomization=True
    )
    loader = dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    output = tft.predict(loader, mode="prediction")
    values = output.detach().cpu().numpy().reshape(-1)

    if len(values) != 1:
        raise ValueError(
            f"Expected one TFT prediction, got {len(values)}. "
            "Check the TFT prediction horizon and dataset configuration."
        )

    return max(0.0, float(values[0]))

st.title("Dharali Rainfall & Cloudburst Prediction")
st.caption("TFT + XGBoost weighted ensemble")

try:
    df = get_weather()
    df["boundary_layer_height"] = (
        df["boundary_layer_height"].ffill().bfill().fillna(0)
    )

    features = create_features(df)

    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    history = features[features["time"] <= now].dropna().reset_index(drop=True)

    if history.empty:
        raise ValueError("No valid recent weather observations available.")

    latest = history.iloc[-1]
    X_latest = pd.DataFrame([latest[feature_cols]], columns=feature_cols)

    
    for col in feature_cols:
        X_latest[col] = pd.to_numeric(
            X_latest[col]
            .astype(str)
            .str.strip()
            .str.strip("[]"),
            errors="coerce"
        )

    if X_latest.isna().any().any():
        bad_cols = X_latest.columns[X_latest.isna().any()].tolist()
        raise ValueError(f"Invalid model input features: {bad_cols}")

    X_latest = clean_model_input(X_latest)

    xgb_prediction = max(
        0.0, float(np.asarray(xgb.predict(X_latest)).reshape(-1)[0])
    )

    tft_prediction = predict_tft(history, tft_params)

    prediction = (
        TFT_WEIGHT * tft_prediction
        + XGB_WEIGHT * xgb_prediction
    )

    st.header("Current Weather")
    cols = st.columns(4)
    cols[0].metric("Temperature", f"{latest['temperature_2m']:.1f} °C")
    cols[1].metric("Humidity", f"{latest['relative_humidity_2m']:.0f}%")
    cols[2].metric("Pressure", f"{latest['surface_pressure']:.0f} hPa")
    cols[3].metric("Wind Speed", f"{latest['wind_speed_10m']:.1f} km/h")

    cols = st.columns(4)
    cols[0].metric("Cloud Cover", f"{latest['cloud_cover']:.0f}%")
    cols[1].metric("Dew Point", f"{latest['dew_point_2m']:.1f} °C")
    cols[2].metric("Wind Gust", f"{latest['wind_gusts_10m']:.1f} km/h")
    cols[3].metric("Current Rain", f"{latest['precipitation']:.2f} mm")

    st.divider()
    st.header("Next-Hour Forecast")

    cols = st.columns(3)
    cols[0].metric("Ensemble Prediction", f"{prediction:.2f} mm/h")
    cols[1].metric("TFT Prediction", f"{tft_prediction:.2f} mm/h")
    cols[2].metric("XGBoost Prediction", f"{xgb_prediction:.2f} mm/h")

    recent_rain = history["precipitation"].tail(5).to_numpy()
    continuous_heavy = (
        len(recent_rain) == 5 and np.all(recent_rain > 30)
    )
    cloudburst = prediction > 100 or continuous_heavy

    if cloudburst:
        st.error("CLOUDBURST ALERT")
    else:
        st.success("No Cloudburst Predicted")

    st.write(
        f"**100 mm/h threshold:** "
        f"{'Exceeded' if prediction > 100 else 'Not exceeded'}"
    )
    st.write(
        f"**30 mm/h sustained condition:** "
        f"{'Detected' if continuous_heavy else 'Not detected'}"
    )
    
    st.divider()
    st.header("Explainable AI — Why This Prediction?")
    st.subheader("XGBoost Feature Contributions")

    try:
        contributions, fig, baseline = explain_xgboost(
            xgb, X_latest
        )

        st.write(f"XGBoost baseline prediction: **{baseline:.3f} mm/h**")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Most Influential Features")
        st.dataframe(
            contributions[
                ["Feature", "Current Value", "SHAP Contribution"]
            ].head(10).round(4),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive contributions increase the model prediction "
            "relative to its baseline; negative contributions decrease it. "
            "These explain XGBoost, not the entire weighted ensemble."
        )
        st.subheader("Top 5 Features Influencing Rainfall Prediction")

        top5 = contributions.head(5).copy()

        top5["Effect"] = top5["SHAP Contribution"].apply(
            lambda x: "Increases prediction" if x > 0
            else "Decreases prediction" if x < 0
            else "No effect"
        )

        st.dataframe(
            top5[["Feature", "Current Value", "SHAP Contribution", "Effect"]],
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.error(f"XAI explanation failed: {e}")

except Exception as e:
    st.error(f"Unable to generate prediction: {e}")