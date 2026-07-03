import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, model_name: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    scores = {
        "rmse": round(float(rmse), 4),
        "mae":  round(float(mae),  4),
        "r2":   round(float(r2),   4),
    }

    label = f"[{model_name}]" if model_name else "[Model]"
    logger.info(f"{label} RMSE: {scores['rmse']}  MAE: {scores['mae']}  R2: {scores['r2']}")
    print(f"  {label} RMSE: {scores['rmse']}  MAE: {scores['mae']}  R2: {scores['r2']}")

    return scores