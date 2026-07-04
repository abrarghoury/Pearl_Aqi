"""
export_to_csv.py
Exports MongoDB collections to CSV files — read-only, no writes.
Run: python export_to_csv.py
"""

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

MONGODB_URI    = os.getenv("MONGODB_URI")
MONGODB_DB     = os.getenv("MONGODB_DB_NAME", "aqi_db")
OUTPUT_DIR     = Path("exports")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_db():
    client = MongoClient(MONGODB_URI)
    return client[MONGODB_DB]


def export_collection(db, collection_name: str, filename: str):
    col     = db[collection_name]
    records = list(col.find({}, {"_id": 0}))

    if not records:
        print(f"  [{collection_name}] No records found — skipping")
        return

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"  [{collection_name}] {len(df)} rows → {out_path}")


def main():
    print("Connecting to MongoDB Atlas...")
    db = get_db()
    print(f"Connected to: {MONGODB_DB}\n")

    print("Exporting collections...")

    # Raw hourly readings
    export_collection(db, "raw_readings",  "raw_readings.csv")

    # Daily features (main training data)
    export_collection(db, "feature_store", "feature_store.csv")

    # Stored predictions history
    export_collection(db, "predictions",   "predictions.csv")

    print(f"\nDone — CSV files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()