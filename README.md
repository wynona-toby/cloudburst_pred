# Explainable AI for Rainfall & Cloudburst Prediction

An AI-based rainfall prediction system for **Dharali, Uttarakhand**, using historical ERA5 weather data from Open-Meteo. The system combines time-series forecasting with machine learning and provides feature-level explanations using SHAP.

## Features

- Historical weather data from Open-Meteo ERA5
- Weather feature engineering and time-series analysis
- Rainfall prediction using TFT and XGBoost
- Weighted ensemble: **85% TFT + 15% XGBoost**
- SHAP-based explanations for XGBoost predictions
- Interactive Streamlit dashboard

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, PyTorch Forecasting, SHAP, Streamlit.

## Project Structure

```text
├── app.py                  # Streamlit application
├── feature_engineering.py  # Weather feature processing
├── xai.py                  # SHAP explanations
├── openrouter.py           # OpenRouter integration
└── models/                 # Trained model artifacts
```

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure any required environment variables in `.env`.

```bash
OPENROUTER_API_KEY=<your-key-here>
```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

## Dataset

Historical hourly weather data for Dharali, Uttarakhand (2020–2026), retrieved from the Open-Meteo Historical Weather API using ERA5 reanalysis.
