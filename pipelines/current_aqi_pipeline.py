"""
pipelines/current_aqi_pipeline.py
Lightweight hourly pipeline — fetches ONLY the current-hour AQI reading
and saves it for the dashboard's "Current AQI" display.

This is intentionally separate from feature_pipeline.py:
  - feature_pipeline.py runs once a day, builds complete-day features,
    and is what the Day1/Day2/Day3 models are trained/predicted on.
  - current_aqi_pipeline.py runs every hour (per project spec: "the
    feature script every hour"), and only updates the live "Current AQI"
    number shown on the dashboard. It never touches feature_store,
    lag/rolling features, or model training in any way.
"""

import sys

from config.settings import settings
from ingestion.fetch_openmeteo import fetch_current_aqi
from features.feature_store import save_current_aqi

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run():
    print("=" * 55)
    print("CURRENT AQI PIPELINE STARTED (hourly)")
    print("=" * 55)

    settings.validate()

    print("\n[Step 1] Fetching current-hour AQI ...")
    current = fetch_current_aqi()

    if current["aqi"] is None:
        print("  No current-hour AQI available yet this run — skipping save.")
    else:
        print(f"  Current AQI: {current['aqi']:.1f} at {current['timestamp']}")
        print("\n[Step 2] Saving current AQI ...")
        save_current_aqi(current["aqi"], current["timestamp"])

    print("\n" + "=" * 55)
    print("CURRENT AQI PIPELINE COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    run()
