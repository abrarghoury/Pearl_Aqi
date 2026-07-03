import logging
import sys
from datetime import datetime

from config.settings import settings
from features.feature_store import load_training_data
from models.registry import save_model
from models.train import _get_feature_cols, train_best_model

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _fill_for_predict(df, feature_cols):
    """
    Mirrors models.train._fill_feature_nulls() but applied to the full df
    right before a .predict() call, since the model was trained on
    mean-filled data. Without this, columns like fc_* (NaN for every
    backfill row) cause sklearn estimators (Ridge especially) to crash
    with "Input X contains NaN" when predicting on the raw df.
    """
    X = df[feature_cols].copy()
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)
    return X


def run():
    """
    Daily training pipeline.
    5 models trained per target, winner selected by CV RMSE.
    All features used — no feature selection
    (feature selection hurt Day2/Day3 performance).

    Recursive strategy:
      Day1 trains on all base features
      Day2 trains with pred_day1 as extra feature
      Day3 trains with pred_day1 + pred_day2 as extra features

    Models saved to MongoDB GridFS with dynamic versioning.
    """
    pipeline_start = datetime.utcnow()

    print("=" * 55)
    print("TRAINING PIPELINE STARTED")
    print("=" * 55)
    logger.info("Training pipeline started")

    settings.validate()

    # Load daily training data
    print("\n[Step 1] Loading training data ...")
    df = load_training_data()
    print(f"  Daily rows loaded: {len(df)}")
    logger.info(f"Loaded {len(df)} daily training rows")

    # Get base feature columns
    base_feature_cols = _get_feature_cols(df, settings.TARGET_DAY1)
    print(f"  Base features available: {len(base_feature_cols)}")

    # ── Day 1 ────────────────────────────────────────────
    print("\n[Step 2] Training Day1 model ...")
    logger.info("Training Day1 model")
    day1_start = datetime.utcnow()

    model_day1, scores_day1 = train_best_model(
        df,
        settings.TARGET_DAY1,
        training_start=day1_start,
    )
    save_model(model_day1, settings.TARGET_DAY1, scores_day1)

    # Generate Day1 predictions on full dataset
    # Use exact features model was trained on — fill NaNs first
    # (model was trained on mean-filled X, predict() needs matching shape/no-NaN)
    day1_features   = scores_day1["features"]
    X_day1          = _fill_for_predict(df, day1_features)
    df["pred_day1"] = model_day1.predict(X_day1)
    logger.info(
        f"Day1 pred mean: {df['pred_day1'].mean():.2f}  "
        f"std: {df['pred_day1'].std():.2f}"
    )

    # ── Day 2 ────────────────────────────────────────────
    print("\n[Step 3] Training Day2 model (+ pred_day1) ...")
    logger.info("Training Day2 model")
    day2_start = datetime.utcnow()

    model_day2, scores_day2 = train_best_model(
        df,
        settings.TARGET_DAY2,
        extra_features=["pred_day1"],
        training_start=day2_start,
    )
    save_model(model_day2, settings.TARGET_DAY2, scores_day2)

    # Generate Day2 predictions using its trained features — fill NaNs first
    day2_features   = scores_day2["features"]
    X_day2          = _fill_for_predict(df, day2_features)
    df["pred_day2"] = model_day2.predict(X_day2)
    logger.info(
        f"Day2 pred mean: {df['pred_day2'].mean():.2f}  "
        f"std: {df['pred_day2'].std():.2f}"
    )

    # ── Day 3 ────────────────────────────────────────────
    print("\n[Step 4] Training Day3 model (+ pred_day1 + pred_day2) ...")
    logger.info("Training Day3 model")
    day3_start = datetime.utcnow()

    model_day3, scores_day3 = train_best_model(
        df,
        settings.TARGET_DAY3,
        extra_features=["pred_day1", "pred_day2"],
        training_start=day3_start,
    )
    save_model(model_day3, settings.TARGET_DAY3, scores_day3)

    total_duration = (datetime.utcnow() - pipeline_start).total_seconds()

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TRAINING PIPELINE COMPLETE")
    print(f"  Day1 : {scores_day1['model_name']:20s}"
          f"  CV RMSE {scores_day1['cv_rmse']:6.4f}"
          f"  Test R2 {scores_day1['r2']:6.4f}")
    print(f"  Day2 : {scores_day2['model_name']:20s}"
          f"  CV RMSE {scores_day2['cv_rmse']:6.4f}"
          f"  Test R2 {scores_day2['r2']:6.4f}")
    print(f"  Day3 : {scores_day3['model_name']:20s}"
          f"  CV RMSE {scores_day3['cv_rmse']:6.4f}"
          f"  Test R2 {scores_day3['r2']:6.4f}")
    print(f"  Total time : {total_duration:.1f}s")
    print("=" * 55)

    logger.info(
        f"Complete in {total_duration:.1f}s | "
        f"Day1 CV:{scores_day1['cv_rmse']} R2:{scores_day1['r2']} | "
        f"Day2 CV:{scores_day2['cv_rmse']} R2:{scores_day2['r2']} | "
        f"Day3 CV:{scores_day3['cv_rmse']} R2:{scores_day3['r2']}"
    )


if __name__ == "__main__":
    run()