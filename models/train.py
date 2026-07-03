import logging
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config.settings import settings
from models.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def _get_feature_cols(df: pd.DataFrame, target_col: str) -> list:
    all_targets = [
        settings.TARGET_DAY1,
        settings.TARGET_DAY2,
        settings.TARGET_DAY3,
    ]

    exclude = set(all_targets) | {
        "date", "timestamp", "ingested_at",
        "city", "source",
        "pred_day1", "pred_day2",
    }

    feature_cols = [
        c for c in settings.FEATURE_COLS
        if c in df.columns
        and c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    missing = [c for c in settings.FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing features for {target_col}: {missing}")
        print(f"  Warning: {len(missing)} features missing: {missing}")

    return feature_cols


def _check_leakage(X: pd.DataFrame, target_col: str):
    all_targets = [
        settings.TARGET_DAY1,
        settings.TARGET_DAY2,
        settings.TARGET_DAY3,
    ]

    for t in all_targets:
        if t in X.columns:
            raise ValueError(
                f"LEAKAGE DETECTED: target '{t}' found in features "
                f"while training for '{target_col}'. "
                "Fix FEATURE_COLS in settings.py."
            )

    suspicious = [
        c for c in X.columns
        if any(k in c for k in ["day1", "day2", "day3"])
        and not c.startswith("pred_")
        and not c.startswith("fc_")
    ]
    if suspicious:
        logger.warning(f"Suspicious columns: {suspicious}")
        print(f"  WARNING: Suspicious columns: {suspicious}")

    print(f"  Leakage check passed for: {target_col}")
    logger.info(f"Leakage check passed for {target_col}")


def _fill_feature_nulls(X: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Fills NaNs in the feature matrix with column means.

    Needed mainly for forecast columns (fc_*_d1/d2/d3, fc_pressure_drop_d1):
    these are NaN for every historical/backfill row (forecast weather only
    exists for "today" in daily-mode runs), and sklearn's RandomForest,
    GradientBoosting, and Ridge all throw on NaN input (unlike XGBoost/
    LightGBM which tolerate it natively). Mean-fill keeps all 5 models
    trainable without dropping rows or columns.

    Columns that are fully NaN (mean itself is NaN) are filled with 0
    as a safe fallback.
    """
    null_counts = X.isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        print(f"  [Train] Filling NaNs in {len(cols_with_nulls)} columns for {target_col}:")
        for col, cnt in cols_with_nulls.items():
            print(f"    {col}: {cnt} NaNs")

    col_means = X.mean(numeric_only=True)
    X = X.fillna(col_means)

    # Fallback for columns that were fully NaN (mean = NaN too)
    X = X.fillna(0)

    return X


def _time_split(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:]
    )


def _cv_score(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 3,
) -> float:
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    rmses = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        X_tr = X_train.iloc[tr_idx]
        X_te = X_train.iloc[te_idx]
        y_tr = y_train.iloc[tr_idx]
        y_te = y_train.iloc[te_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(np.mean((y_te.values - pred) ** 2)))
        rmses.append(rmse)
        logger.debug(f"Fold {fold+1} RMSE: {rmse:.4f}")

    mean_cv = float(np.mean(rmses))
    logger.debug(f"Mean CV RMSE: {mean_cv:.4f}")
    return mean_cv


def _tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.8},
    ]

    best_rmse, best_params = float("inf"), param_grid[0]

    for params in param_grid:
        m = XGBRegressor(
            **params,
            colsample_bytree = 0.8,
            min_child_weight = 3,
            reg_alpha        = 0.5,
            reg_lambda       = 2.0,
            random_state     = 42,
            verbosity        = 0,
        )
        rmse = _cv_score(m, X_train, y_train)
        logger.info(f"XGB {params} CV RMSE {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params

    logger.info(f"Best XGB: {best_params} CV RMSE {best_rmse:.4f}")
    return XGBRegressor(
        **best_params,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.5,
        reg_lambda       = 2.0,
        random_state     = 42,
        verbosity        = 0,
    ), best_rmse


def _tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    param_grid = [
        {"n_estimators": 200, "max_depth": 6,  "min_samples_leaf": 8,  "max_features": 0.7},
        {"n_estimators": 300, "max_depth": 8,  "min_samples_leaf": 6,  "max_features": 0.7},
        {"n_estimators": 200, "max_depth": 6,  "min_samples_leaf": 10, "max_features": 0.8},
        {"n_estimators": 300, "max_depth": 5,  "min_samples_leaf": 8,  "max_features": 0.6},
    ]

    best_rmse, best_params = float("inf"), param_grid[0]

    for params in param_grid:
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        rmse = _cv_score(m, X_train, y_train)
        logger.info(f"RF {params} CV RMSE {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params

    logger.info(f"Best RF: {best_params} CV RMSE {best_rmse:.4f}")
    return RandomForestRegressor(**best_params, random_state=42, n_jobs=-1), best_rmse


def _tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    param_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
    ]

    best_rmse, best_params = float("inf"), param_grid[0]

    for params in param_grid:
        m = GradientBoostingRegressor(**params, random_state=42)
        rmse = _cv_score(m, X_train, y_train)
        logger.info(f"GB {params} CV RMSE {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params

    logger.info(f"Best GB: {best_params} CV RMSE {best_rmse:.4f}")
    return GradientBoostingRegressor(**best_params, random_state=42), best_rmse


def _tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    param_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 20, "max_depth": 5},
        {"n_estimators": 100, "learning_rate": 0.10, "num_leaves": 15, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 25, "max_depth": 5},
    ]

    best_rmse, best_params = float("inf"), param_grid[0]

    for params in param_grid:
        m = LGBMRegressor(
            **params,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_alpha        = 0.5,
            reg_lambda       = 2.0,
            random_state     = 42,
            verbose          = -1,
        )
        rmse = _cv_score(m, X_train, y_train)
        logger.info(f"LGB {params} CV RMSE {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params

    logger.info(f"Best LGB: {best_params} CV RMSE {best_rmse:.4f}")
    return LGBMRegressor(
        **best_params,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.5,
        reg_lambda       = 2.0,
        random_state     = 42,
        verbose          = -1,
    ), best_rmse


def _get_ridge_cv(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0)),
    ])
    cv_rmse = _cv_score(model, X_train, y_train)
    logger.info(f"Ridge CV RMSE {cv_rmse:.4f}")
    return model, cv_rmse


def train_best_model(
    df: pd.DataFrame,
    target_col: str,
    extra_features: list = None,
    training_start: datetime = None,
) -> tuple:
    """
    Train 5 regression models on daily aggregated data.
    Models: XGBoost, RandomForest, GradientBoosting, LightGBM, Ridge.

    NOTE: forecast (fc_*) features are NaN for backfill rows (forecast
    weather only exists for "today"). _fill_feature_nulls() mean-fills
    these before training so RandomForest/GradientBoosting/Ridge don't
    crash on NaN input. XGBoost/LightGBM would tolerate NaN natively,
    but we fill for all 5 so the CV comparison stays fair (same input
    data for every model).

    Flow:
      1. Build feature matrix from settings.FEATURE_COLS
      2. Add recursive prediction features (pred_day1/pred_day2)
      3. Fill NaNs (forecast cols on backfill rows, etc.)
      4. Leakage check
      5. Chronological 80/20 train/test split
      6. Tune all 5 models via TimeSeriesSplit CV (3 folds)
      7. Select winner by lowest CV RMSE
      8. Retrain winner on full training set
      9. Evaluate all models on test set (transparency only)
     10. Return winner + scores dict
    """
    if training_start is None:
        training_start = datetime.utcnow()

    logger.info(f"Training for target: {target_col}")
    print(f"\n  Target : {target_col}")

    # Build feature matrix
    feature_cols = _get_feature_cols(df, target_col)
    X = df[feature_cols].copy()

    # Add recursive prediction features safely
    if extra_features:
        all_targets = [
            settings.TARGET_DAY1,
            settings.TARGET_DAY2,
            settings.TARGET_DAY3,
        ]
        for feat in extra_features:
            if feat in all_targets:
                raise ValueError(
                    f"LEAKAGE: '{feat}' is a real target. "
                    "Only pass pred_day1 or pred_day2."
                )
            if feat in df.columns:
                X[feat] = df[feat].values
            else:
                logger.warning(f"Extra feature '{feat}' not found — skipping")
                print(f"  Warning: '{feat}' not found — skipping")

    y = df[target_col]

    # Fill NaNs (forecast columns etc.) before anything else touches X
    X = _fill_feature_nulls(X, target_col)

    # Leakage check — hard stop
    _check_leakage(X, target_col)

    # Chronological split
    X_train, X_test, y_train, y_test = _time_split(X, y)

    print(f"  Train  : {len(X_train)} rows  |  Test : {len(X_test)} rows")
    print(f"  Features: {len(X.columns)} total")

    if len(X_train) < 50:
        raise RuntimeError(
            f"Training set too small: {len(X_train)} rows. "
            "Run backfill pipeline first."
        )

    # Tune all 5 models via CV
    print("  Tuning XGBoost ...")
    xgb_model,   xgb_cv   = _tune_xgboost(X_train, y_train)

    print("  Tuning RandomForest ...")
    rf_model,    rf_cv    = _tune_random_forest(X_train, y_train)

    print("  Tuning GradientBoosting ...")
    gb_model,    gb_cv    = _tune_gradient_boosting(X_train, y_train)

    print("  Tuning LightGBM ...")
    lgb_model,   lgb_cv   = _tune_lightgbm(X_train, y_train)

    print("  Evaluating Ridge ...")
    ridge_model, ridge_cv = _get_ridge_cv(X_train, y_train)

    candidates = {
        "XGBoost":          (xgb_model,   xgb_cv),
        "RandomForest":     (rf_model,    rf_cv),
        "GradientBoosting": (gb_model,    gb_cv),
        "LightGBM":         (lgb_model,   lgb_cv),
        "Ridge":            (ridge_model, ridge_cv),
    }

    # Show CV comparison
    print("\n  CV RMSE comparison (winner selected from here):")
    min_cv = min(v[1] for v in candidates.values())
    for name, (_, cv_rmse) in candidates.items():
        marker = " <-- winner" if cv_rmse == min_cv else ""
        print(f"    {name:20s} CV RMSE {cv_rmse:.4f}{marker}")

    # Select winner by lowest CV RMSE
    best_name    = min(candidates, key=lambda k: candidates[k][1])
    best_model   = candidates[best_name][0]
    best_cv_rmse = candidates[best_name][1]
    logger.info(f"CV winner: {best_name} CV RMSE {best_cv_rmse:.4f}")

    # Retrain winner on full training set
    best_model.fit(X_train, y_train)

    # Evaluate all on test set — transparency only
    print("\n  Test set scores (reporting only):")
    for name, (model, _) in candidates.items():
        model.fit(X_train, y_train)
        evaluate_model(y_test, model.predict(X_test), model_name=name)

    # Final winner scores
    best_model.fit(X_train, y_train)
    y_pred_final = best_model.predict(X_test)
    final_scores = evaluate_model(
        y_test, y_pred_final,
        model_name=f"{best_name} [CV WINNER]"
    )

    duration = (datetime.utcnow() - training_start).total_seconds()

    final_scores["model_name"]                = best_name
    final_scores["cv_rmse"]                   = round(best_cv_rmse, 4)
    final_scores["feature_count"]             = len(X.columns)
    final_scores["features"]                  = list(X.columns)
    final_scores["data_rows_used"]            = len(df)
    final_scores["training_duration_seconds"] = round(duration, 2)

    print(
        f"\n  FINAL: {best_name}  "
        f"CV RMSE {best_cv_rmse:.4f}  "
        f"Test RMSE {final_scores['rmse']}  "
        f"Test R2 {final_scores['r2']}"
    )

    return best_model, final_scores