import time
from datetime import datetime, timedelta

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from config.settings import settings


def _build_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    return openmeteo_requests.Client(session=retry_session)


def _get_date_range():
    today = datetime.utcnow().date()

    if settings.PIPELINE_MODE == "backfill":
        start = today - timedelta(days=settings.BACKFILL_DAYS)
        end   = today - timedelta(days=1)
    else:
        start = today - timedelta(days=3)
        end   = today - timedelta(days=1)

    return str(start), str(end)


def _parse_hourly_response(response, variables: list) -> pd.DataFrame:
    hourly = response.Hourly()

    data = {"timestamp": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )}

    for i, var in enumerate(variables):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data)


def _fetch_air_quality(client, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":   settings.CITY_LAT,
        "longitude":  settings.CITY_LON,
        "hourly":     settings.AQ_VARIABLES,
        "timezone":   settings.TIMEZONE,
        "start_date": start_date,
        "end_date":   end_date,
    }
    responses = client.weather_api(settings.OPENMETEO_AQ_URL, params=params)
    return _parse_hourly_response(responses[0], settings.AQ_VARIABLES)


def _fetch_weather(client, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":   settings.CITY_LAT,
        "longitude":  settings.CITY_LON,
        "hourly":     settings.WEATHER_VARIABLES,
        "timezone":   settings.TIMEZONE,
        "start_date": start_date,
        "end_date":   end_date,
    }
    responses = client.weather_api(settings.OPENMETEO_ARCHIVE_URL, params=params)
    return _parse_hourly_response(responses[0], settings.WEATHER_VARIABLES)


def _fetch_forecast_weather(client) -> pd.DataFrame:
    """
    Fetches NEXT N days of forecast weather (not historical).
    Used to build fc_*_d1/d2/d3 features — known-in-advance signal
    for Day2/Day3 AQI prediction, instead of relying purely on
    recursive AQI predictions which compound error.
    """
    params = {
        "latitude":   settings.CITY_LAT,
        "longitude":  settings.CITY_LON,
        "hourly":     settings.FORECAST_WEATHER_VARIABLES,
        "timezone":   settings.TIMEZONE,
        "forecast_days": settings.FORECAST_DAYS,
    }
    responses = client.weather_api(settings.OPENMETEO_FORECAST_URL, params=params)
    df = _parse_hourly_response(responses[0], settings.FORECAST_WEATHER_VARIABLES)
    df = df.rename(columns=settings.COLUMN_RENAME_MAP)
    return df


def _fetch_in_chunks(
    client,
    fetch_fn,
    start_date: str,
    end_date: str,
    chunk_days: int = 30,
) -> pd.DataFrame:
    all_chunks    = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    final_end     = datetime.strptime(end_date,   "%Y-%m-%d").date()

    while current_start <= final_end:
        current_end = min(
            current_start + timedelta(days=chunk_days - 1),
            final_end
        )
        print(f"  Fetching {current_start} to {current_end} ...")

        try:
            chunk = fetch_fn(client, str(current_start), str(current_end))
            all_chunks.append(chunk)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Warning: chunk failed ({current_start} to {current_end}): {e}")

        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        raise RuntimeError(
            "All API chunks failed — no data fetched. "
            "Check internet connection and OpenMeteo API status."
        )

    return pd.concat(all_chunks, ignore_index=True)


def _aggregate_forecast_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast hourly -> one row per future date with mean values.
    Used to build fc_*_d1/d2/d3 columns relative to "today".
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date

    daily = df.groupby("date").agg(
        temperature = ("temperature", "mean"),
        humidity    = ("humidity",    "mean"),
        wind_speed  = ("wind_speed",  "mean"),
        pressure    = ("pressure",    "mean"),
        precipitation = ("precipitation", "sum"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def fetch_forecast_features() -> dict:
    """
    Returns a flat dict of fc_*_d1/d2/d3 features for the next 3 days,
    to be merged into today's daily feature row.

    Called only in daily mode — backfill has no "forecast" concept,
    historical data covers everything needed there.
    """
    client = _build_client()
    raw    = _fetch_forecast_weather(client)
    daily  = _aggregate_forecast_to_daily(raw)

    today = datetime.utcnow().date()
    features = {}

    for i in range(1, settings.FORECAST_DAYS + 1):
        target_date = today + timedelta(days=i)
        row = daily[daily["date"].dt.date == target_date]

        if row.empty:
            print(f"  [Forecast] Warning: no forecast row for day {i} ({target_date})")
            features[f"fc_temp_d{i}"]     = None
            features[f"fc_humidity_d{i}"] = None
            features[f"fc_wind_d{i}"]     = None
            features[f"fc_pressure_d{i}"] = None
            features[f"fc_precip_d{i}"]   = None
            continue

        r = row.iloc[0]
        features[f"fc_temp_d{i}"]     = float(r["temperature"])
        features[f"fc_humidity_d{i}"] = float(r["humidity"])
        features[f"fc_wind_d{i}"]     = float(r["wind_speed"])
        features[f"fc_pressure_d{i}"] = float(r["pressure"])
        features[f"fc_precip_d{i}"]   = float(r["precipitation"])

    print(f"[Forecast] Built {len(features)} forecast features for next {settings.FORECAST_DAYS} days")
    return features


def fetch_raw_data() -> pd.DataFrame:
    """
    Main function — called by all pipelines.
    Returns hourly merged DataFrame (air quality + weather).

    Backfill mode: fetches full BACKFILL_DAYS in 30-day chunks
    Daily mode:    fetches last 3 days only

    Forecast weather is fetched SEPARATELY via fetch_forecast_features()
    since it's a different shape (future, not historical hourly rows).
    """
    print(f"[Fetch] Mode: {settings.PIPELINE_MODE}")

    client               = _build_client()
    start_date, end_date = _get_date_range()

    print(f"[Fetch] Date range: {start_date} to {end_date}")
    print(f"[Fetch] City: {settings.CITY_NAME} ({settings.CITY_LAT}, {settings.CITY_LON})")

    print("[Fetch] Fetching air quality ...")
    if settings.PIPELINE_MODE == "backfill":
        aq_df = _fetch_in_chunks(client, _fetch_air_quality, start_date, end_date)
    else:
        aq_df = _fetch_air_quality(client, start_date, end_date)

    print("[Fetch] Fetching weather ...")
    if settings.PIPELINE_MODE == "backfill":
        weather_df = _fetch_in_chunks(client, _fetch_weather, start_date, end_date)
    else:
        weather_df = _fetch_weather(client, start_date, end_date)

    print("[Fetch] Merging ...")
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")

    merged = merged.rename(columns=settings.COLUMN_RENAME_MAP)

    merged["city"]        = settings.CITY_NAME
    merged["ingested_at"] = datetime.utcnow()
    merged["source"]      = "openmeteo"

    merged["timestamp"] = merged["timestamp"].dt.tz_localize(None)

    print(f"[Fetch] Done — {len(merged)} rows, {len(merged.columns)} columns")
    print(f"[Fetch] Columns: {list(merged.columns)}")

    return merged