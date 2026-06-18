import os

PROJECT_ID = "project-8e2366a6-d3cc-40ee-9de"
REGION     = "asia-southeast1"

AR_REPO    = "hospital-ml"
IMAGE_NAME = "hospital-ml-pipeline"
IMAGE_TAG  = "latest"                 
IMAGE_URI  = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"
)

DEFAULT_CF_URL = f"https://{REGION}-{PROJECT_ID}.cloudfunctions.net/hospital-inference-pipeline"

CF_INFERENCE_URL = os.getenv("CF_INFERENCE_URL", DEFAULT_CF_URL)

TRAIN_FS_TABLE             = f"{PROJECT_ID}.hospital_feature_store.fs_hospital_weekly"
INFER_FS_TABLE             = f"{PROJECT_ID}.hospital_feature_store.fs_hospital_weekly_demo"
FS_DATASET           = "hospital_feature_store"

RAW_TABLE            = f"{PROJECT_ID}.ml_predictions_dev.vertex_batch_prediction_raw"
FORECAST_TABLE       = f"{PROJECT_ID}.ml_predictions_dev.ml_forecast_results"
OBSERVABILITY_TABLE  = f"{PROJECT_ID}.ml_observability.model_predictions_shap"
DIM_TABLE            = f"{PROJECT_ID}.hospital_dwh_dev.dim_hospital"

BUCKET_NAME  = f"{PROJECT_ID}-hospital-model"
XGB_PREFIX   = "hospital-model/xgboost"
LGB_PREFIX   = "hospital-model/lightgbm"
STAGING_GCS  = f"gs://{BUCKET_NAME}/staging"     

TRAIN_END    = "2023-12-25"
VAL_END      = "2024-09-24"

MACHINE_TRAIN_PIPELINE  = "n1-highmem-8"   # data_prep + train (memory-intensive)
MACHINE_EVALUATE        = "n1-standard-4"  # evaluate_and_register
MACHINE_INFERENCE       = "n1-standard-4"  # inference (BQ I/O + predict)

SERVICE_ACCOUNT = (
    "hospital-ml-pipeline@project-8e2366a6-d3cc-40ee-9de.iam.gserviceaccount.com"
)

GCP_CONN_ID = "google_cloud_default"