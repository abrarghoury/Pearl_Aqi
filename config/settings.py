import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:

    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "aqi_db")

    COLLECTION_RAW: str = "raw_readings"
    COLLECTION_FEATURES: str = "feature_store"
    COLLECTION_MODELS: str = "model_metadata"
    COLLECTION_PREDICTIONS: str = "predictions"

    CITY_NAME: str = os.getenv("CITY_NAME", "Istanbul")
    CITY_LAT: float = float(os.getenv("CITY_LAT", "41.0082"))
    CITY_LON: float = float(os.getenv("CITY_LON", "28.9784"))
    TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Istanbul")

    PIPELINE_MODE: str = os.getenv("PIPELINE_MODE", "daily")
    BACKFILL_DAYS: int = 730  # 2 years — more rows stabilizes Day2/Day3

    OPENMETEO_AQ_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    OPENMETEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPENMETEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPENMETEO_AQ_FORECAST_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

    FORECAST_DAYS: int = 3  # how many future days of weather to pull

    AQ_VARIABLES: list = [
        "pm10", "pm2_5", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide",
        "ozone", "us_aqi",
    ]

    WEATHER_VARIABLES: list = [
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
        "surface_pressure", "precipitation",
    ]

    # Forecast weather variables (subset — what matters most for AQI dispersion)
    FORECAST_WEATHER_VARIABLES: list = [
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
        "surface_pressure", "precipitation",
    ]

    COLUMN_RENAME_MAP: dict = {
        "carbon_monoxide":      "co",
        "nitrogen_dioxide":     "no2",
        "sulphur_dioxide":      "so2",
        "ozone":                "o3",
        "relative_humidity_2m": "humidity",
        "temperature_2m":       "temperature",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_direction",
        "surface_pressure":     "pressure",
        "us_aqi":               "aqi",
    }

    TARGET_DAY1: str = "aqi_day1"
    TARGET_DAY2: str = "aqi_day2"
    TARGET_DAY3: str = "aqi_day3"

    # All candidate features
    FEATURE_COLS: list = [
        # Daily AQI aggregations
        "aqi_mean",
        "aqi_max",
        "aqi_min",
        "aqi_std",
        "aqi_last6h",

        # Pollutants
        "pm2_5_mean",
        "pm10_mean",

        # Weather
        "temp_mean",
        "temp_max",
        "temp_min",
        "humidity_mean",
        "wind_mean",
        "wind_max",
        "pressure_mean",

        # Lag features
        "aqi_lag1d",
        "aqi_lag2d",
        "aqi_lag3d",
        "aqi_lag7d",
        "pm2_5_lag1d",
        "pm2_5_lag2d",

        # Rolling features
        "aqi_roll_mean_3",
        "aqi_roll_mean_7",

        # Trend + diff features
        "aqi_trend_1d",
        "aqi_trend_3d",
        "aqi_diff",
        "aqi_pct_change",

        # Pollutant / weather delta features (NEW)
        "pm2_5_diff",
        "pm10_diff",
        "temp_diff",
        "pressure_diff",
        "wind_diff",
        "humidity_diff",

        # Volatility
        "aqi_std_7d",

        # Time
        "month_sin",
        "month_cos",
        "dow",

        # Forecast weather features (NEW) — known in advance, huge for Day2/Day3
        "fc_temp_d1", "fc_temp_d2", "fc_temp_d3",
        "fc_humidity_d1", "fc_humidity_d2", "fc_humidity_d3",
        "fc_wind_d1", "fc_wind_d2", "fc_wind_d3",
        "fc_pressure_d1", "fc_pressure_d2", "fc_pressure_d3",
        "fc_precip_d1", "fc_precip_d2", "fc_precip_d3",
        "fc_pressure_drop_d1",  # pressure_mean -> fc_pressure_d1, storm signal
    ]

    TOP_K_FEATURES: int = 15

    AQI_CATEGORIES: dict = {
        "Good":                  (0,   50),
        "Moderate":              (51,  100),
        "Unhealthy (Sensitive)": (101, 150),
        "Unhealthy":             (151, 200),
        "Very Unhealthy":        (201, 300),
        "Hazardous":             (301, 500),
    }

    PREDICTION_CONFIDENCE: dict = {
        "aqi_day1": "High",
        "aqi_day2": "Moderate",
        "aqi_day3": "Low",
    }

    AQI_ALERT_THRESHOLD: int = 150

    def validate(self):
        errors = []

        if not self.MONGODB_URI:
            errors.append("MONGODB_URI is not set in .env")

        if self.PIPELINE_MODE not in ("backfill", "daily"):
            errors.append(
                f"PIPELINE_MODE '{self.PIPELINE_MODE}' is invalid. "
                "Use 'backfill' or 'daily'"
            )

        if errors:
            raise EnvironmentError(
                "Fix these config issues:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        print(f"[Config] Mode     : {self.PIPELINE_MODE}")
        print(f"[Config] City     : {self.CITY_NAME} ({self.CITY_LAT}, {self.CITY_LON})")
        print(f"[Config] Database : {self.MONGODB_DB_NAME}")
        print("[Config] All good")


settings = Settings()