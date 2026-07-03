"""
app/predict.py
Central prediction module — loads features, runs models, computes SHAP.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import shap
import streamlit as st
from pymongo import MongoClient

from config.settings import settings
from models.registry import load_model, get_model_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AQI category
# ---------------------------------------------------------------------------

AQI_CATEGORIES = [
    (0,   50,  "Good",                  "#05be05", "#000000"),
    (51,  100, "Moderate",              "#e5e50f", "#000000"),
    (101, 150, "Unhealthy (Sensitive)", "#f07a05", "#ffffff"),
    (151, 200, "Unhealthy",             "#ed0303", "#ffffff"),
    (201, 300, "Very Unhealthy",        "#802b88", "#ffffff"),
    (301, 500, "Hazardous",             "#730222", "#ffffff"),
]


def get_aqi_category(aqi_value: float) -> dict:
    aqi_value = max(0, float(aqi_value))
    for low, high, label, bg_color, text_color in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return {
                "label":      label,
                "bg_color":   bg_color,
                "text_color": text_color,
                "low":        low,
                "high":       high,
            }
    return {
        "label":      "Hazardous",
        "bg_color":   "#7e0023",
        "text_color": "#ffffff",
        "low":        301,
        "high":       500,
    }


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def _get_db():
    client = MongoClient(settings.MONGODB_URI)
    return client[settings.MONGODB_DB_NAME]


def _load_feature_history(days: int = 30) -> pd.DataFrame:
    db     = _get_db()
    col    = db[settings.COLLECTION_FEATURES]
    cutoff = datetime.utcnow() - timedelta(days=days)

    records = list(
        col.find({"date": {"$gte": cutoff}}, {"_id": 0}).sort("date", 1)
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_latest_feature_row() -> pd.DataFrame:
    db  = _get_db()
    col = db[settings.COLLECTION_FEATURES]

    records = list(col.find({}, {"_id": 0}).sort("date", -1).limit(1))
    if not records:
        raise RuntimeError(
            "No features found in MongoDB — run the feature pipeline first."
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_past_predictions(days: int = 14) -> pd.DataFrame:
    db     = _get_db()
    col    = db[settings.COLLECTION_PREDICTIONS]
    cutoff = datetime.utcnow() - timedelta(days=days)

    records = list(
        col.find({"prediction_date": {"$gte": cutoff}}, {"_id": 0})
        .sort("prediction_date", 1)
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def _build_feature_matrix(row: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = pd.DataFrame(index=row.index)

    for col in feature_cols:
        if col in row.columns:
            X[col] = row[col].values
        else:
            logger.warning("Feature '%s' missing — filling with 0", col)
            X[col] = 0.0

    col_means = X.mean(numeric_only=True)
    X = X.fillna(col_means)
    X = X.fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

# Class names (as strings, so we don't need xgboost/lightgbm/catboost
# installed just to reference their classes) that SHAP's TreeExplainer
# can handle directly.
_TREE_MODEL_HINTS = (
    "RandomForest", "GradientBoosting", "ExtraTrees", "DecisionTree",
    "XGB", "LGBM", "LightGBM", "CatBoost", "HistGradientBoosting",
)

# Linear models SHAP's LinearExplainer can handle directly.
_LINEAR_MODEL_HINTS = (
    "Ridge", "Lasso", "ElasticNet", "LinearRegression", "SGDRegressor",
    "BayesianRidge", "ARDRegression",
)


def _unwrap_estimator(model):
    """
    If model is an sklearn Pipeline, return its final estimator
    (regardless of what the step is named). Otherwise return model as-is.
    """
    from sklearn.pipeline import Pipeline as SklearnPipeline

    if isinstance(model, SklearnPipeline):
        return model.steps[-1][1]
    return model


def _classify_estimator(estimator) -> str:
    name = type(estimator).__name__
    if any(hint in name for hint in _TREE_MODEL_HINTS):
        return "tree"
    if any(hint in name for hint in _LINEAR_MODEL_HINTS):
        return "linear"
    return "unknown"


def _get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Picks the fastest correct SHAP explainer for the given model.

    - Tree-based estimators (raw, or wrapped in a Pipeline) -> TreeExplainer
    - Linear estimators (raw, or wrapped in a Pipeline)      -> LinearExplainer
    - Anything else (SVR, MLP, stacked/voting models, custom
      wrappers, unknown Pipeline contents, etc.)              -> generic
      model-agnostic Explainer built on model.predict (works
      for ANY model, just slower).

    Returns (explainer, kind) where kind is "tree" | "linear" | "generic".
    """
    from sklearn.pipeline import Pipeline as SklearnPipeline

    final_estimator = _unwrap_estimator(model)
    kind = _classify_estimator(final_estimator)

    if kind == "tree":
        # TreeExplainer needs the raw tree estimator, not the Pipeline.
        # If there's a Pipeline in front of it, its earlier steps must
        # already be pure passthrough/no-op for feature space to line up
        # (most feature stores hand in already-numeric features). If any
        # preprocessing changes shape, this will raise and we fall back.
        return shap.TreeExplainer(final_estimator), "tree"

    if kind == "linear":
        masker = shap.maskers.Independent(X_background)
        return shap.LinearExplainer(model, masker), "linear"

    # Generic fallback — works for literally any predict() function.
    # Slower (permutation-based) but never silently fails just because
    # the model type isn't one of the two fast-paths above.
    background = shap.sample(X_background, min(50, len(X_background))) \
        if len(X_background) > 0 else X_background
    masker = shap.maskers.Independent(background)
    return shap.Explainer(model.predict, masker), "generic"


def _compute_shap_values(
    model,
    X_input: pd.DataFrame,
    X_background: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Returns (shap_values_1d, error_reason).
    error_reason is None on success, otherwise a short human-readable
    string describing why SHAP could not be computed — this gets shown
    directly in the dashboard instead of a generic message.
    """
    # Attempt 1: fast-path explainer chosen by model type
    try:
        explainer, kind = _get_shap_explainer(model, X_background)
        shap_values = explainer(X_input)
        values_1d = np.array(shap_values.values).reshape(-1)[: X_input.shape[1]]
        logger.info(
            "SHAP computed via %s explainer — top feature: %s (%.2f)",
            kind,
            X_input.columns[np.abs(values_1d).argmax()],
            values_1d[np.abs(values_1d).argmax()],
        )
        return values_1d, None
    except Exception as exc:
        logger.warning(
            "Fast-path SHAP explainer failed for model %s: %s",
            type(model).__name__, exc,
        )

    # Attempt 2: guaranteed-to-work generic fallback (permutation-based
    # explainer using model.predict directly). Slower, but works
    # regardless of model internals.
    try:
        background = shap.sample(X_background, min(50, len(X_background))) \
            if len(X_background) > 0 else X_background
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(model.predict, masker)
        shap_values = explainer(X_input)
        values_1d = np.array(shap_values.values).reshape(-1)[: X_input.shape[1]]
        logger.info(
            "SHAP computed via generic fallback explainer for model %s",
            type(model).__name__,
        )
        return values_1d, None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error(
            "SHAP computation failed for model %s (both fast-path and "
            "generic fallback): %s",
            type(model).__name__, reason, exc_info=True,
        )
        return None, reason


def _get_base_value(model, X_background: pd.DataFrame) -> Optional[float]:
    """
    expected_value can be a scalar or array depending on model/SHAP version.
    Always extract a single float. Falls back to the mean of the model's
    predictions over the background set if the explainer's own
    expected_value isn't available.
    """
    try:
        explainer, _ = _get_shap_explainer(model, X_background)
        ev = explainer.expected_value
        if hasattr(ev, "__len__"):
            return float(ev[0])
        return float(ev)
    except Exception as exc:
        logger.warning("Could not get base value from explainer: %s", exc)

    # Fallback — mean prediction over background rows is a reasonable
    # stand-in for the SHAP "base value" (average model output).
    try:
        if len(X_background) == 0:
            return None
        preds = model.predict(X_background)
        return float(np.mean(preds))
    except Exception as exc:
        logger.warning("Could not compute fallback base value: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prediction persistence
# ---------------------------------------------------------------------------

def _save_prediction(prediction_date: datetime, results: dict):
    db  = _get_db()
    col = db[settings.COLLECTION_PREDICTIONS]

    document = {
        "prediction_date": prediction_date,
        "city":            settings.CITY_NAME,
        "predicted_at":    datetime.utcnow(),
        "day1": {"aqi": results["day1"]["aqi"], "label": results["day1"]["category"]["label"]},
        "day2": {"aqi": results["day2"]["aqi"], "label": results["day2"]["category"]["label"]},
        "day3": {"aqi": results["day3"]["aqi"], "label": results["day3"]["category"]["label"]},
    }

    col.update_one(
        {"prediction_date": prediction_date},
        {"$set": document},
        upsert=True,
    )
    logger.info("Prediction saved for %s", prediction_date.date())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def run_prediction() -> dict:
    """
    Full prediction pipeline cached for 1 hour.
    Returns single dict consumed by all dashboard pages.
    """
    logger.info("Running prediction pipeline")

    latest_row      = _load_latest_feature_row()
    feature_history = _load_feature_history(days=30)
    past_preds      = _load_past_predictions(days=14)

    latest_date = latest_row["date"].iloc[0]
    latest_aqi  = float(latest_row["aqi_mean"].iloc[0]) if "aqi_mean" in latest_row.columns else 0.0

    shap_background_raw = feature_history.copy() if not feature_history.empty else latest_row.copy()

    # Load models
    model_day1 = load_model(settings.TARGET_DAY1)
    model_day2 = load_model(settings.TARGET_DAY2)
    model_day3 = load_model(settings.TARGET_DAY3)
    metadata   = get_model_metadata()

    meta_by_target = {m["target"]: m for m in metadata}
    features_day1  = meta_by_target.get(settings.TARGET_DAY1, {}).get("features", [])
    features_day2  = meta_by_target.get(settings.TARGET_DAY2, {}).get("features", [])
    features_day3  = meta_by_target.get(settings.TARGET_DAY3, {}).get("features", [])

    logger.info(
        "Model types — Day1: %s | Day2: %s | Day3: %s",
        type(model_day1).__name__, type(model_day2).__name__, type(model_day3).__name__,
    )

    # Feature matrices
    X_day1          = _build_feature_matrix(latest_row, features_day1)
    pred_day1_value = float(model_day1.predict(X_day1)[0])

    day2_row = latest_row.copy()
    if "pred_day1" in features_day2:
        day2_row["pred_day1"] = pred_day1_value
    X_day2          = _build_feature_matrix(day2_row, features_day2)
    pred_day2_value = float(model_day2.predict(X_day2)[0])

    day3_row = latest_row.copy()
    if "pred_day1" in features_day3:
        day3_row["pred_day1"] = pred_day1_value
    if "pred_day2" in features_day3:
        day3_row["pred_day2"] = pred_day2_value
    X_day3          = _build_feature_matrix(day3_row, features_day3)
    pred_day3_value = float(model_day3.predict(X_day3)[0])

    pred_day1_value = max(0.0, pred_day1_value)
    pred_day2_value = max(0.0, pred_day2_value)
    pred_day3_value = max(0.0, pred_day3_value)

    logger.info(
        "Predictions — Day1: %.1f  Day2: %.1f  Day3: %.1f",
        pred_day1_value, pred_day2_value, pred_day3_value,
    )

    # SHAP backgrounds
    shap_bg_day1 = _build_feature_matrix(shap_background_raw, features_day1)
    shap_bg_day2 = _build_feature_matrix(shap_background_raw, features_day2)
    shap_bg_day3 = _build_feature_matrix(shap_background_raw, features_day3)

    shap_day1, shap_err_day1 = _compute_shap_values(model_day1, X_day1, shap_bg_day1)
    shap_day2, shap_err_day2 = _compute_shap_values(model_day2, X_day2, shap_bg_day2)
    shap_day3, shap_err_day3 = _compute_shap_values(model_day3, X_day3, shap_bg_day3)

    base_day1 = _get_base_value(model_day1, shap_bg_day1)
    base_day2 = _get_base_value(model_day2, shap_bg_day2)
    base_day3 = _get_base_value(model_day3, shap_bg_day3)

    results = {
        "prediction_date":  latest_date + timedelta(days=1),
        "latest_date":      latest_date,
        "latest_aqi":       latest_aqi,

        "day1": {
            "aqi":           pred_day1_value,
            "category":      get_aqi_category(pred_day1_value),
            "shap_values":   shap_day1,
            "shap_features": list(X_day1.columns),
            "base_value":    base_day1,
            "shap_error":    shap_err_day1,
        },
        "day2": {
            "aqi":           pred_day2_value,
            "category":      get_aqi_category(pred_day2_value),
            "shap_values":   shap_day2,
            "shap_features": list(X_day2.columns),
            "base_value":    base_day2,
            "shap_error":    shap_err_day2,
        },
        "day3": {
            "aqi":           pred_day3_value,
            "category":      get_aqi_category(pred_day3_value),
            "shap_values":   shap_day3,
            "shap_features": list(X_day3.columns),
            "base_value":    base_day3,
            "shap_error":    shap_err_day3,
        },

        "model_metadata":   metadata,
        "feature_history":  feature_history,
        "past_predictions": past_preds,
    }

    try:
        _save_prediction(results["prediction_date"], results)
    except Exception as exc:
        logger.warning("Could not save prediction: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Helpers used by dashboard pages
# ---------------------------------------------------------------------------

def get_top_shap_features(
    shap_values: Optional[np.ndarray],
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    if shap_values is None or len(shap_values) == 0:
        return pd.DataFrame(columns=["feature", "shap_value", "abs_shap"])

    df = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": shap_values,
        "abs_shap":   np.abs(shap_values),
    })
    return df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def build_waterfall_data(
    shap_values: Optional[np.ndarray],
    feature_names: list,
    base_value: Optional[float],
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    if shap_values is None or base_value is None:
        return None

    top_df          = get_top_shap_features(shap_values, feature_names, top_n=top_n)
    shown_features  = set(top_df["feature"].tolist())
    all_df          = pd.DataFrame({"feature": feature_names, "shap_value": shap_values})
    other_sum       = all_df[~all_df["feature"].isin(shown_features)]["shap_value"].sum()

    rows = top_df[["feature", "shap_value"]].copy()
    if abs(other_sum) > 0.01:
        rows = pd.concat(
            [rows, pd.DataFrame([{"feature": "other features", "shap_value": other_sum}])],
            ignore_index=True,
        )

    rows["running_total"] = base_value + rows["shap_value"].cumsum()
    rows["color"]         = rows["shap_value"].apply(lambda v: "#ffffff" if v > 0 else "#ffffff")
    return rows


def generate_shap_insight(
    shap_values: Optional[np.ndarray],
    feature_names: list,
    day_label: str = "today",
) -> str:
    if shap_values is None or len(shap_values) == 0:
        return "SHAP explanation unavailable for this model."

    df = get_top_shap_features(shap_values, feature_names, top_n=3)
    if df.empty:
        return "No dominant features identified."

    parts = []
    for _, row in df.iterrows():
        direction = "pushed the forecast up by" if row["shap_value"] > 0 else "reduced it by"
        parts.append(f"{row['feature']} {direction} {abs(row['shap_value']):.1f} AQI points")

    return f"For {day_label}: " + ", and ".join(parts) + "."