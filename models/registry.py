import logging
from datetime import datetime
from io import BytesIO

import joblib
import gridfs
from pymongo import MongoClient

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_db():
    client = MongoClient(settings.MONGODB_URI)
    return client[settings.MONGODB_DB_NAME]


def save_model(model, target_col: str, scores: dict):
    """
    Save trained model to MongoDB GridFS.
    Keeps last 2 versions — older ones auto-deleted.
    Dynamic versioning: v1, v2, v3 ...
    scores dict must contain: model_name, cv_rmse, rmse, mae, r2,
                              feature_count, features, data_rows_used,
                              training_duration_seconds
    """
    db  = _get_db()
    fs  = gridfs.GridFS(db)
    col = db[settings.COLLECTION_MODELS]

    # Serialize model to bytes — no local disk needed
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename      = (
        f"{target_col}_{scores.get('model_name', 'model')}_{timestamp_str}.joblib"
    )

    # Save to GridFS
    file_id = fs.put(buffer, filename=filename)
    logger.info(f"Saved to GridFS: {filename} id={file_id}")

    # Keep only last 2 versions — delete older GridFS files + metadata
    all_versions = list(
        col.find({"target": target_col}).sort("version_num", -1)
    )
    if len(all_versions) >= 2:
        for old in all_versions[2:]:
            try:
                fs.delete(old["gridfs_file_id"])
                logger.info(f"Deleted old GridFS file: {old['gridfs_file_id']}")
            except Exception as e:
                logger.warning(f"Could not delete old model file: {e}")
            col.delete_one({"_id": old["_id"]})

    # Get next version number
    latest = col.find(
        {"target": target_col}
    ).sort("version_num", -1).limit(1)
    last_num = 0
    for doc in latest:
        last_num = doc.get("version_num", 0)
    new_version_num = last_num + 1

    # Save metadata
    col.insert_one({
        "target":                    target_col,
        "model_name":                scores.get("model_name"),
        "version":                   f"v{new_version_num}",
        "version_num":               new_version_num,
        "status":                    "active",
        "gridfs_file_id":            file_id,
        "gridfs_filename":           filename,
        "cv_rmse":                   scores.get("cv_rmse"),
        "test_rmse":                 scores.get("rmse"),
        "test_mae":                  scores.get("mae"),
        "test_r2":                   scores.get("r2"),
        "feature_count":             scores.get("feature_count", 0),
        "features":                  scores.get("features", []),
        "data_rows_used":            scores.get("data_rows_used", 0),
        "trained_at":                datetime.utcnow(),
        "training_duration_seconds": scores.get("training_duration_seconds", 0),
        # Confidence level for dashboard display
        "confidence":                settings.PREDICTION_CONFIDENCE.get(target_col, "Unknown"),
    })

    print(
        f"  [Registry] {target_col} v{new_version_num} "
        f"({scores.get('model_name')}) "
        f"CV RMSE {scores.get('cv_rmse')}  "
        f"Test R2 {scores.get('r2')}"
    )
    logger.info(f"Registered {target_col} v{new_version_num}")


def load_model(target_col: str):
    """
    Load latest active model from GridFS.
    Raises FileNotFoundError if model not found.
    """
    db  = _get_db()
    fs  = gridfs.GridFS(db)
    col = db[settings.COLLECTION_MODELS]

    doc = col.find_one(
        {"target": target_col, "status": "active"},
        sort=[("version_num", -1)],
    )

    if not doc:
        raise FileNotFoundError(
            f"No active model for '{target_col}' — run training pipeline first."
        )

    grid_file = fs.get(doc["gridfs_file_id"])
    model     = joblib.load(BytesIO(grid_file.read()))

    print(
        f"  [Registry] Loaded {target_col} "
        f"{doc['version']} ({doc['model_name']}) from GridFS"
    )
    logger.info(f"Loaded {target_col} {doc['version']} from GridFS")

    return model


def get_model_metadata() -> list:
    """
    Return metadata for all 3 latest active models.
    Used by dashboard to display model info and confidence levels.
    """
    db  = _get_db()
    col = db[settings.COLLECTION_MODELS]

    targets = [
        settings.TARGET_DAY1,
        settings.TARGET_DAY2,
        settings.TARGET_DAY3,
    ]

    results = []
    for target in targets:
        doc = col.find_one(
            {"target": target, "status": "active"},
            sort=[("version_num", -1)],
        )
        if doc:
            doc.pop("_id", None)
            doc.pop("gridfs_file_id", None)
            results.append(doc)

    return results