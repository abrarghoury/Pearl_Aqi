import numpy as np
import pandas as pd

from config.settings import settings

REQUIRED_COLUMNS = ["aqi", "pm2_5", "pm10", "temperature", "humidity"]

# Istanbul realistic ranges — updated from Karachi
VALID_RANGES = {
    "aqi":           (0,   500),
    "pm2_5":         (0,   500),
    "pm10":          (0,   600),
    "co":            (0,   15000),
    "no2":           (0,   500),
    "so2":           (0,   500),
    "o3":            (0,   500),
    "temperature":   (-15, 45),   # Istanbul range: cold winters, hot summers
    "humidity":      (0,   100),
    "wind_speed":    (0,   150),
    "pressure":      (900, 1100),
    "precipitation": (0,   200),
}


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    removed = before - len(df)
    if removed > 0:
        print(f"  [Clean] Removed {removed} duplicate rows")
    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Values outside valid range replaced with NaN
    # Interpolation fills them in next step
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range > 0:
            print(f"  [Clean] {col}: {out_of_range} outliers set to NaN")
            df[col] = df[col].where(
                (df[col] >= low) & (df[col] <= high), other=np.nan
            )
    return df


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Linear interpolation for short gaps
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)

    # Forward fill for remaining edge gaps
    df[numeric_cols] = df[numeric_cols].ffill()

    # Backward fill for gaps at very start
    df[numeric_cols] = df[numeric_cols].bfill()

    return df


def _drop_required_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    removed = before - len(df)
    if removed > 0:
        print(f"  [Clean] Dropped {removed} rows with unfillable nulls")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """c
    Clean hourly raw data fetched from OpenMeteo.
    Called before daily aggregation in compute_features.py.
    Steps: remove duplicates → cap outliers → fill nulls → drop unfillable
    """
    print(f"[Clean] Starting with {len(df)} rows")

    df = _remove_duplicates(df)
    df = _cap_outliers(df)
    df = _fill_nulls(df)
    df = _drop_required_nulls(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[Clean] Done — {len(df)} rows remaining")
    return df