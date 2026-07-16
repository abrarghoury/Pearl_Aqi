import sys

from config.settings import settings
from features.clean import clean_raw_data
from features.compute_features import compute_features
from features.feature_store import (
    get_feature_count,
    save_raw,
    save_features,
    load_recent_raw,
    save_current_aqi,   # 🆕 ADDED
)
from ingestion.fetch_openmeteo import fetch_raw_data, fetch_forecast_features, fetch_current_aqi  # 🆕 ADDED fetch_current_aqi

import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# How many days of history to pull back from MongoDB so that lag/rolling
# features (e.g. aqi_lag7d, aqi_roll_mean_7, aqi_std_7d) have enough
# lookback to compute on. Daily fetch only brings in the last 3 days,
# which is never enough on its own.
HISTORY_LOOKBACK_DAYS = 14


def run():
    """
    Runs every day via GitHub Actions (not every hour).
    Fetches last 3 days of data -> cleans -> combines with stored
    history -> aggregates to daily.
    compute_targets=False because we have no future AQI data yet.
    Upsert handles duplicates -- safe to run multiple times.

    NEW: also fetches forward-looking forecast weather (next 3 days)
    via fetch_forecast_features() and passes it into compute_features()
    so fc_*_d1/d2/d3 columns get attached to today's row.

    NEW: pulls back HISTORY_LOOKBACK_DAYS of already-saved hourly data
    from MongoDB and combines it with the freshly fetched 3 days before
    computing features. Without this, lag/rolling features that need
    more than 3 days of lookback (e.g. 7-day lag) would always be NaN
    and every row would get dropped, forever — not just during a
    temporary cold-start period.
    """
    print("=" * 55)
    print("FEATURE PIPELINE STARTED")
    print("=" * 55)

    settings.validate()

    # Step 1 — fetch last 3 days hourly data
    # Extra days ensure no boundary gaps
    print("\n[Step 1] Fetching latest raw data ...")
    df_raw = fetch_raw_data()
    print(f"  Fetched: {len(df_raw)} hourly rows")

    # Step 2 — clean hourly data
    print("\n[Step 2] Cleaning ...")
    df_clean = clean_raw_data(df_raw)
    print(f"  After clean: {len(df_clean)} hourly rows")

    # Step 3 — save raw hourly data
    print("\n[Step 3] Saving raw data ...")
    save_raw(df_clean)

    # Step 4 — fetch forecast weather (next 3 days) — NEW
    print("\n[Step 4] Fetching forecast weather (next 3 days) ...")
    try:
        forecast_dict = fetch_forecast_features()
        print(f"  Forecast features: {forecast_dict}")
    except Exception as e:
        print(f"  Warning: forecast fetch failed ({e}) — continuing without it")
        forecast_dict = None

    # 🆕 ADDED — Step 4b: fetch + save today's live hourly AQI, used ONLY
    # for the dashboard's "Current AQI" display. Completely separate from
    # the feature/training data below — cannot affect aqi_mean, lag,
    # rolling, or any model feature.
    print("\n[Step 4b] Fetching current-hour AQI (for dashboard display only) ...")
    try:
        current = fetch_current_aqi()
        if current["aqi"] is not None:
            save_current_aqi(current["aqi"], current["timestamp"])
    except Exception as e:
        print(f"  Warning: current AQI fetch failed ({e}) — continuing without it")

    # Step 5 — combine freshly fetched data with stored history
    # so lag/rolling features (7-day lag, 7-day rolling mean/std) have
    # enough lookback. Without this, they'd always be NaN in daily mode.
    print(f"\n[Step 5] Loading last {HISTORY_LOOKBACK_DAYS} days of history from MongoDB ...")
    df_history = load_recent_raw(days=HISTORY_LOOKBACK_DAYS)

    if df_history.empty:
        df_combined = df_clean
    else:
        df_combined = (
            pd.concat([df_history, df_clean], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    print(f"  Combined dataset: {len(df_combined)} hourly rows")

    # Step 6 — aggregate to daily + compute features
    # No targets — future AQI not available in daily mode
    print("\n[Step 6] Computing daily features (no targets) ...")
    df_features = compute_features(
        df_combined,
        compute_targets=False,
        forecast_dict=forecast_dict,
    )
    print(f"  Daily rows computed: {len(df_features)}")

    # Step 7 — save/update daily features
    # Upsert on date — existing days updated, new days inserted
    print("\n[Step 7] Saving features ...")
    save_features(df_features)

    total = get_feature_count()

    print("\n" + "=" * 55)
    print("FEATURE PIPELINE COMPLETE")
    print(f"  Daily rows processed : {len(df_features)}")
    print(f"  Total days in MongoDB: {total}")
    print("=" * 55)


if __name__ == "__main__":
    run()
