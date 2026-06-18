import os

PROJECT_ID  = os.getenv("PROJECT_ID",  "project-8e2366a6-d3cc-40ee-9de")
REGION      = os.getenv("REGION",      "asia-southeast1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "project-8e2366a6-d3cc-40ee-9de-hospital-model")

INFER_FS_TABLE = os.getenv(
    "INFER_FS_TABLE",
    f"{PROJECT_ID}.hospital_feature_store.fs_hospital_weekly_demo",
)
RAW_TABLE = os.getenv(
    "RAW_TABLE",
    f"{PROJECT_ID}.ml_predictions_dev.vertex_batch_prediction_raw",
)
FORECAST_TABLE = os.getenv(
    "FORECAST_TABLE",
    f"{PROJECT_ID}.ml_predictions_dev.ml_forecast_results",
)
OBSERVABILITY_TABLE = os.getenv(
    "OBSERVABILITY_TABLE",
    f"{PROJECT_ID}.ml_observability.model_predictions_shap",
)

DIM_TABLE = os.getenv(
    "DIM_TABLE",
    f"{PROJECT_ID}.hospital_dwh_dev.dim_hospital",
)

XGB_PREFIX = os.getenv("XGB_PREFIX", "hospital-model/xgboost")
LGB_PREFIX = os.getenv("LGB_PREFIX", "hospital-model/lightgbm")   # Empty string = LightGBM disabled

DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "35"))
MIN_RAW_ROWS          = int(os.getenv("MIN_RAW_ROWS", "1"))