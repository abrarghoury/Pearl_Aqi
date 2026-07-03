import numpy as np
import pandas as pd

from config.settings import settings

REQUIRED_HOURLY_COLS = ["aqi", "pm2_5", "pm10", "temperature", "humidity"]


def _validate_hourly_input(df: pd.DataFrame):
    missing = [c for c in REQUIRED_HOURLY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' missing from hourly data.")

    if len(df) < 24:
        raise ValueError(f"Only {len(df)} rows — need at least 24.")


def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date

    daily = df.groupby("date").agg(
        aqi_mean      = ("aqi",         "mean"),
        aqi_max       = ("aqi",         "max"),
        aqi_min       = ("aqi",         "min"),
        aqi_std       = ("aqi",         "std"),
        aqi_last6h    = ("aqi",         lambda x: x.iloc[-6:].mean() if len(x) >= 6 else x.mean()),
        pm2_5_mean    = ("pm2_5",       "mean"),
        pm10_mean     = ("pm10",        "mean"),
        temp_mean     = ("temperature", "mean"),
        temp_max      = ("temperature", "max"),
        temp_min      = ("temperature", "min"),
        humidity_mean = ("humidity",    "mean"),
        wind_mean     = ("wind_speed",  "mean"),
        wind_max      = ("wind_speed",  "max"),
        pressure_mean = ("pressure",    "mean"),
        precip_sum    = ("precipitation", "sum") if "precipitation" in df.columns else ("aqi", "size"),
    ).reset_index()

    daily["date"]    = pd.to_datetime(daily["date"])
    daily            = daily.sort_values("date").reset_index(drop=True)
    daily["aqi_std"] = daily["aqi_std"].fillna(0)

    return daily


def _add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["aqi_lag1d"]   = daily["aqi_mean"].shift(1)
    daily["aqi_lag2d"]   = daily["aqi_mean"].shift(2)
    daily["aqi_lag3d"]   = daily["aqi_mean"].shift(3)
    daily["aqi_lag7d"]   = daily["aqi_mean"].shift(7)
    daily["pm2_5_lag1d"] = daily["pm2_5_mean"].shift(1)
    daily["pm2_5_lag2d"] = daily["pm2_5_mean"].shift(2)
    return daily


def _add_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["aqi_roll_mean_3"] = (
        daily["aqi_mean"].shift(1).rolling(window=3, min_periods=2).mean()
    )
    daily["aqi_roll_mean_7"] = (
        daily["aqi_mean"].shift(1).rolling(window=7, min_periods=4).mean()
    )
    return daily


def _add_trend_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["aqi_trend_1d"]   = daily["aqi_mean"].shift(1) - daily["aqi_mean"].shift(2)
    daily["aqi_trend_3d"]   = daily["aqi_mean"].shift(1) - daily["aqi_mean"].shift(4)
    daily["aqi_diff"]        = daily["aqi_mean"].shift(1).diff()
    daily["aqi_pct_change"]  = daily["aqi_mean"].shift(1).pct_change() * 100
    return daily


def _add_delta_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["pm2_5_diff"]   = daily["pm2_5_mean"].shift(1).diff()
    daily["pm10_diff"]    = daily["pm10_mean"].shift(1).diff()
    daily["temp_diff"]    = daily["temp_mean"].shift(1).diff()
    daily["pressure_diff"] = daily["pressure_mean"].shift(1).diff()
    daily["wind_diff"]    = daily["wind_mean"].shift(1).diff()
    daily["humidity_diff"] = daily["humidity_mean"].shift(1).diff()
    return daily


def _add_volatility_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["aqi_std_7d"] = (
        daily["aqi_mean"].shift(1)
        .rolling(window=7, min_periods=3)
        .std()
        .fillna(0)
    )
    return daily


def _add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["month_sin"] = np.sin(2 * np.pi * daily["date"].dt.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["date"].dt.month / 12)
    daily["dow"]       = daily["date"].dt.dayofweek
    return daily


def _add_forecast_features(daily: pd.DataFrame, forecast_dict: dict = None) -> pd.DataFrame:
    """
    REAL forecast features — not mean/zero-filled.

    For every row EXCEPT the most recent one, we already know the
    actual future weather (it's historical/archive data), so we use
    shift(-1)/-2/-3 on the ACTUAL daily weather columns as a
    "perfect forecast" proxy. This gives genuine, varying signal
    during training — unlike NaN->0 fill which had zero variance.

    For the LAST row only (today, in daily-mode live runs), tomorrow's
    actual weather doesn't exist yet — that's where the real Open-Meteo
    forecast API (forecast_dict) fills in fc_*_d1/d2/d3.

    This means: training learns from real historical "what actually
    happened next" data, and live prediction uses a real forecast API
    call for the same features — consistent meaning, no dead columns.
    """
    daily["fc_temp_d1"]     = daily["temp_mean"].shift(-1)
    daily["fc_temp_d2"]     = daily["temp_mean"].shift(-2)
    daily["fc_temp_d3"]     = daily["temp_mean"].shift(-3)

    daily["fc_humidity_d1"] = daily["humidity_mean"].shift(-1)
    daily["fc_humidity_d2"] = daily["humidity_mean"].shift(-2)
    daily["fc_humidity_d3"] = daily["humidity_mean"].shift(-3)

    daily["fc_wind_d1"]     = daily["wind_mean"].shift(-1)
    daily["fc_wind_d2"]     = daily["wind_mean"].shift(-2)
    daily["fc_wind_d3"]     = daily["wind_mean"].shift(-3)

    daily["fc_pressure_d1"] = daily["pressure_mean"].shift(-1)
    daily["fc_pressure_d2"] = daily["pressure_mean"].shift(-2)
    daily["fc_pressure_d3"] = daily["pressure_mean"].shift(-3)

    daily["fc_precip_d1"]   = daily["precip_sum"].shift(-1)
    daily["fc_precip_d2"]   = daily["precip_sum"].shift(-2)
    daily["fc_precip_d3"]   = daily["precip_sum"].shift(-3)

    # Overwrite ONLY the last row with real forecast API data, if provided
    # (live daily-mode run — tomorrow's actual weather isn't known yet)
    if forecast_dict:
        last_idx = daily.index[-1]
        col_map = {
            "fc_temp_d1": "fc_temp_d1", "fc_temp_d2": "fc_temp_d2", "fc_temp_d3": "fc_temp_d3",
            "fc_humidity_d1": "fc_humidity_d1", "fc_humidity_d2": "fc_humidity_d2", "fc_humidity_d3": "fc_humidity_d3",
            "fc_wind_d1": "fc_wind_d1", "fc_wind_d2": "fc_wind_d2", "fc_wind_d3": "fc_wind_d3",
            "fc_pressure_d1": "fc_pressure_d1", "fc_pressure_d2": "fc_pressure_d2", "fc_pressure_d3": "fc_pressure_d3",
            "fc_precip_d1": "fc_precip_d1", "fc_precip_d2": "fc_precip_d2", "fc_precip_d3": "fc_precip_d3",
        }
        for col in col_map:
            val = forecast_dict.get(col)
            if val is not None:
                daily.loc[last_idx, col] = val

    daily["fc_pressure_drop_d1"] = daily["pressure_mean"] - daily["fc_pressure_d1"]

    return daily


def _add_targets(daily: pd.DataFrame) -> pd.DataFrame:
    daily[settings.TARGET_DAY1] = daily["aqi_mean"].shift(-1)
    daily[settings.TARGET_DAY2] = daily["aqi_mean"].shift(-2)
    daily[settings.TARGET_DAY3] = daily["aqi_mean"].shift(-3)
    return daily


def _check_leakage(daily: pd.DataFrame, feature_cols: list):
    all_targets = [
        settings.TARGET_DAY1,
        settings.TARGET_DAY2,
        settings.TARGET_DAY3,
    ]

    for target in all_targets:
        if target in feature_cols:
            raise ValueError(
                f"LEAKAGE: target '{target}' found in features. "
                "Fix FEATURE_COLS in settings.py."
            )

    suspicious = [
        c for c in feature_cols
        if any(k in c for k in ["day1", "day2", "day3"])
    ]
    if suspicious:
        raise ValueError(
            f"LEAKAGE: suspicious columns in features: {suspicious}"
        )

    print("  [Features] Leakage check passed")


def _drop_rows_with_null_targets(daily: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        settings.TARGET_DAY1,
        settings.TARGET_DAY2,
        settings.TARGET_DAY3,
    ]
    before  = len(daily)
    daily   = daily.dropna(subset=target_cols)
    dropped = before - len(daily)
    if dropped > 0:
        print(f"  [Features] Dropped {dropped} rows — null targets at end (expected)")
    return daily


def _drop_rows_with_null_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Note: fc_*_d3 (and target columns) both use shift(-3), so the last
    3 rows of a backfill run will have NaN forecast features too — same
    rows already getting dropped by _drop_rows_with_null_targets when
    compute_targets=True. So this no longer needs a special-case for
    forecast columns; they're real (non-constant) for all kept rows.
    """
    feature_cols_present = [
        c for c in settings.FEATURE_COLS if c in daily.columns
    ]
    before  = len(daily)
    daily   = daily.dropna(subset=feature_cols_present)
    dropped = before - len(daily)
    if dropped > 0:
        print(f"  [Features] Dropped {dropped} rows — null features at start/end (expected)")
    return daily


def compute_features(
    df: pd.DataFrame,
    compute_targets: bool = True,
    forecast_dict: dict = None,
) -> pd.DataFrame:
    """
    Convert hourly raw data to daily feature DataFrame.

    Steps:
      1. Validate hourly input
      2. Aggregate hourly to daily
      3. Lag features (1d, 2d, 3d, 7d)
      4. Rolling mean (3d, 7d)
      5. Trend + diff features
      6. Delta features (pollutant/weather day-over-day)
      7. Volatility (7d std)
      8. Time features (month cyclical, dow)
      9. Forecast features — real shift(-N) on actual data for history,
         real forecast API for the live/last row
     10. Targets (next 1/2/3 day avg AQI)
     11. Leakage check
     12. Drop null rows

    forecast_dict: output of ingestion.fetch_openmeteo.fetch_forecast_features(),
                   only passed in daily-mode pipeline (None during backfill,
                   where shift(-N) on real future data is used for ALL rows
                   including what would be "today").

    Returns daily DataFrame — one row per calendar day.
    """
    print(f"[Features] Starting with {len(df)} hourly rows")

    _validate_hourly_input(df)

    daily = _aggregate_to_daily(df)
    print(f"[Features] Aggregated to {len(daily)} daily rows")

    daily = _add_lag_features(daily)
    daily = _add_rolling_features(daily)
    daily = _add_trend_features(daily)
    daily = _add_delta_features(daily)
    daily = _add_volatility_features(daily)
    daily = _add_time_features(daily)
    daily = _add_forecast_features(daily, forecast_dict=forecast_dict)

    if compute_targets:
        daily = _add_targets(daily)

    feature_cols = [c for c in settings.FEATURE_COLS if c in daily.columns]
    _check_leakage(daily, feature_cols)

    if compute_targets:
        daily = _drop_rows_with_null_targets(daily)

    daily = _drop_rows_with_null_features(daily)
    daily = daily.reset_index(drop=True)

    print(f"[Features] Done — {len(daily)} daily rows, {len(daily.columns)} columns")
    print(f"[Features] Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    return daily