import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 3, 6, 12, 24]:
        df[f"rain_lag_{lag}h"] = df["precipitation"].shift(lag)

    for window in [3, 6, 24]:
        df[f"rain_rolling_{window}h"] = df["precipitation"].shift(1).rolling(window).sum()

    weather_cols = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "wind_speed_10m", "wind_gusts_10m",
        "total_column_integrated_water_vapour", "boundary_layer_height",
        "vapour_pressure_deficit", "wet_bulb_temperature_2m"
    ]

    for col in weather_cols:
        df[f"{col}_change_1h"] = df[col].diff()

    df["wind_direction_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["wind_direction_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))

    df.drop(columns=["wind_direction_10m"], inplace=True)

    return df