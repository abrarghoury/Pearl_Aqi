import pandas as pd
from pymongo import MongoClient, ASCENDING, ReplaceOne
from config.settings import settings


def _get_collection(collection_name: str):
    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]
    return db[collection_name]


def _ensure_indexes():
    # Raw collection — unique on timestamp (hourly)
    raw_col = _get_collection(settings.COLLECTION_RAW)
    raw_col.create_index([("timestamp", ASCENDING)], unique=True)

    # Feature store — unique on date (daily level now)
    # One row per day — date is the natural key
    feat_col = _get_collection(settings.COLLECTION_FEATURES)
    feat_col.create_index([("date", ASCENDING)], unique=True)


def _bulk_upsert(col, records: list, key_field: str) -> tuple:
    """
    Bulk upsert using key_field as match condition.
    raw_readings  → key_field = "timestamp"
    feature_store → key_field = "date"
    Safe to rerun — no duplicates created.
    """
    if not records:
        # Nothing to write — avoid pymongo's "No operations to execute" crash
        return 0, 0

    operations = [
        ReplaceOne(
            {key_field: record[key_field]},
            record,
            upsert=True
        )
        for record in records
    ]
    result = col.bulk_write(operations, ordered=False)
    return result.upserted_count, result.modified_count


def save_raw(df: pd.DataFrame) -> int:
    """
    Save hourly raw data to raw_readings collection.
    Key field: timestamp — one row per hour.
    """
    if df.empty:
        print("[FeatureStore] Raw: no rows to save — skipping.")
        return 0

    _ensure_indexes()
    col     = _get_collection(settings.COLLECTION_RAW)
    records = df.to_dict(orient="records")
    inserted, updated = _bulk_upsert(col, records, key_field="timestamp")
    print(f"[FeatureStore] Raw: {inserted} new, {updated} updated — {len(records)} total")
    return inserted


def save_features(df: pd.DataFrame) -> int:
    """
    Save daily aggregated features to feature_store collection.
    Key field: date — one row per day.
    Safe to rerun — existing dates get updated, new dates get inserted.

    NOTE: In the early days of a fresh pipeline (or whenever there isn't
    enough historical raw data yet to compute rolling/lag features), the
    feature engineering step may legitimately produce zero valid rows.
    This is expected during cold-start and should not crash the run —
    it will resolve itself once enough daily history accumulates.
    """
    if df.empty:
        print(
            "[FeatureStore] Features: no valid rows to save this run "
            "(likely not enough historical data yet for rolling features) — skipping."
        )
        return 0

    _ensure_indexes()
    col     = _get_collection(settings.COLLECTION_FEATURES)
    records = df.to_dict(orient="records")
    inserted, updated = _bulk_upsert(col, records, key_field="date")
    print(f"[FeatureStore] Features: {inserted} new, {updated} updated — {len(records)} total")
    return inserted


def load_recent_raw(days: int) -> pd.DataFrame:
    """
    Load the last `days` days of hourly raw data from MongoDB
    (raw_readings collection).

    Needed because daily-mode fetch_raw_data() only pulls the last
    3 days from the API each run. Lag/rolling features (e.g. 7-day
    lag, 7-day rolling mean/std) need more history than that — so we
    pull the accumulated history back out of MongoDB and combine it
    with the freshly fetched data before computing features.
    """
    col    = _get_collection(settings.COLLECTION_RAW)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)

    records = list(
        col.find({"timestamp": {"$gte": cutoff}}, {"_id": 0})
        .sort("timestamp", 1)
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"[FeatureStore] Loaded {len(df)} historical hourly rows (last {days} days) for feature computation")
    return df


def load_training_data() -> pd.DataFrame:
    """
    Load all daily rows that have all 3 target columns.
    Used by training_pipeline.py.
    Returns DataFrame sorted by date ascending.
    """
    col = _get_collection(settings.COLLECTION_FEATURES)
    query = {
        settings.TARGET_DAY1: {"$exists": True},
        settings.TARGET_DAY2: {"$exists": True},
        settings.TARGET_DAY3: {"$exists": True},
    }
    records = list(col.find(query, {"_id": 0}))

    if not records:
        raise RuntimeError(
            "No training data found — run backfill pipeline first"
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[FeatureStore] Loaded {len(df)} training rows from MongoDB")
    return df


def load_latest_features() -> pd.DataFrame:
    """
    Load most recent daily row for prediction.
    Used by predict.py in the web app.
    Returns single row DataFrame with all feature columns.
    """
    col = _get_collection(settings.COLLECTION_FEATURES)
    records = list(
        col.find({}, {"_id": 0})
        .sort("date", -1)
        .limit(1)
    )

    if not records:
        raise RuntimeError(
            "No features found — run feature pipeline first"
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[FeatureStore] Loaded latest daily row: {df['date'].iloc[0].date()}")
    return df


def get_feature_count() -> int:
    col = _get_collection(settings.COLLECTION_FEATURES)
    return col.count_documents({})


# 🆕 ADDED — small, separate collection just for the live "Current AQI"
# dashboard number. Never touched by training/feature pipeline logic.
def save_current_aqi(aqi_value: float, timestamp) -> None:
    """
    Saves the single latest hourly AQI reading to its own small
    collection — separate from feature_store, never used in training.
    """
    if aqi_value is None:
        print("[FeatureStore] Current AQI: nothing to save — skipping.")
        return

    col = _get_collection("current_aqi")
    col.update_one(
        {"city": settings.CITY_NAME},
        {"$set": {
            "city":       settings.CITY_NAME,
            "aqi":        aqi_value,
            "timestamp":  timestamp,
            "updated_at": pd.Timestamp.utcnow().tz_localize(None),
        }},
        upsert=True,
    )
    print(f"[FeatureStore] Current AQI saved: {aqi_value:.1f}")


# 🆕 ADDED — reads back the live current AQI for the dashboard.
def load_current_aqi() -> dict:
    """Loads the latest live hourly AQI reading for dashboard display."""
    col = _get_collection("current_aqi")
    doc = col.find_one({"city": settings.CITY_NAME}, {"_id": 0})
    return doc or {}
