import sys

from config.settings import settings
from features.clean import clean_raw_data
from features.compute_features import compute_features
from features.feature_store import save_raw, save_features
from ingestion.fetch_openmeteo import fetch_raw_data

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run():
    """
    Run ONCE to populate MongoDB with 1 year of historical data.
    Fetches hourly data → cleans → aggregates to daily → saves.
    After this, feature_pipeline.py takes over via GitHub Actions.
    """
    print("=" * 55)
    print("BACKFILL PIPELINE STARTED")
    print("=" * 55)

    # Force backfill mode — ignore whatever .env says
    settings.PIPELINE_MODE = "backfill"
    settings.validate()

    # Step 1 — fetch 1 year hourly raw data in 30-day chunks
    print("\n[Step 1] Fetching 1 year of raw data ...")
    df_raw = fetch_raw_data()
    print(f"  Fetched: {len(df_raw)} hourly rows")

    # Step 2 — clean hourly data
    print("\n[Step 2] Cleaning raw data ...")
    df_clean = clean_raw_data(df_raw)
    print(f"  After clean: {len(df_clean)} hourly rows")

    # Step 3 — save hourly raw to MongoDB (backup + audit trail)
    print("\n[Step 3] Saving raw hourly data to MongoDB ...")
    save_raw(df_clean)

    # Step 4 — aggregate to daily + compute features + targets
    # compute_targets=True because we have full year of future data
    print("\n[Step 4] Computing daily features + targets ...")
    df_features = compute_features(df_clean, compute_targets=True)
    print(f"  After aggregation: {len(df_features)} daily rows")

    # Step 5 — save daily features to MongoDB
    print("\n[Step 5] Saving daily features to MongoDB ...")
    save_features(df_features)

    print("\n" + "=" * 55)
    print("BACKFILL COMPLETE")
    print(f"  Raw hourly rows saved : {len(df_clean)}")
    print(f"  Daily feature rows    : {len(df_features)}")
    print(f"  Date range            : "
          f"{df_features['date'].min().date()} to "
          f"{df_features['date'].max().date()}")
    print("=" * 55)
    print("\nNext step: set PIPELINE_MODE=daily in .env")
    print("Then run: python -m pipelines.training_pipeline")


if __name__ == "__main__":
    run()