import sys

from config.settings import settings
from features.clean import clean_raw_data
from features.compute_features import compute_features
from features.feature_store import get_feature_count, save_raw, save_features
from ingestion.fetch_openmeteo import fetch_raw_data, fetch_forecast_features

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run():
    """
    Runs every day via GitHub Actions (not every hour).
    Fetches last 3 days of data -> cleans -> aggregates to daily.
    compute_targets=False because we have no future AQI data yet.
    Upsert handles duplicates -- safe to run multiple times.

    NEW: also fetches forward-looking forecast weather (next 3 days)
    via fetch_forecast_features() and passes it into compute_features()
    so fc_*_d1/d2/d3 columns get attached to today's row.
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

    # Step 5 — aggregate to daily + compute features
    # No targets — future AQI not available in daily mode
    print("\n[Step 5] Computing daily features (no targets) ...")
    df_features = compute_features(
        df_clean,
        compute_targets=False,
        forecast_dict=forecast_dict,
    )
    print(f"  Daily rows computed: {len(df_features)}")

    # Step 6 — save/update daily features
    # Upsert on date — existing days updated, new days inserted
    print("\n[Step 6] Saving features ...")
    save_features(df_features)

    total = get_feature_count()

    print("\n" + "=" * 55)
    print("FEATURE PIPELINE COMPLETE")
    print(f"  Daily rows processed : {len(df_features)}")
    print(f"  Total days in MongoDB: {total}")
    print("=" * 55)


if __name__ == "__main__":
    run()