"""
pipelines/prediction_pipeline.py

Runs the prediction pipeline (Day1/Day2/Day3 AQI forecast + SHAP) once
and saves the result to the predictions collection — independent of the
Streamlit dashboard.

Why this exists: app.predict.run_prediction() is wrapped in Streamlit's
@st.cache_data, which only executes when someone opens the dashboard.
If the app isn't visited for a while, no new documents get written to
the predictions collection, and the Live Accuracy page's sample count
stops growing even though the feature/training pipelines keep running
on their own schedule. This script calls the same underlying logic
directly, on a GitHub Actions schedule, so predictions accumulate daily
regardless of dashboard traffic.
"""

import sys

from config.settings import settings
from app.predict import _generate_predictions

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run():
    print("=" * 55)
    print("PREDICTION PIPELINE STARTED")
    print("=" * 55)

    settings.validate()

    results = _generate_predictions()

    print(f"\n  Prediction date : {results['prediction_date'].date()}")
    print(f"  Day1 : {results['day1']['aqi']:.1f} ({results['day1']['category']['label']})")
    print(f"  Day2 : {results['day2']['aqi']:.1f} ({results['day2']['category']['label']})")
    print(f"  Day3 : {results['day3']['aqi']:.1f} ({results['day3']['category']['label']})")

    print("\n" + "=" * 55)
    print("PREDICTION PIPELINE COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    run()
