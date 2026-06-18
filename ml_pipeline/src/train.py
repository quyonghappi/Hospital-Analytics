"""
train.py — Production training entrypoint for hospital utilization ML pipeline.

Reads pre-processed artifacts từ data_prep.py (train/val parquet + preprocessor.pkl +
feature_metadata.json), fits XGBoost và LightGBM (regression + classification heads),
builds SHAP TreeExplainer cho mỗi model, lưu toàn bộ artifacts vào GCS.

Flow:
  data_prep.py  →  [GCS: preprocessor.pkl + feature_metadata.json + split parquets]
  train.py      →  [GCS: model_reg.pkl + model_cls.pkl + explainer.pkl]  (per model tag)
  evaluate_and_register.py  →  Vertex AI Model Registry

GCS layout sau khi train hoàn tất:
  gs://<bucket>/hospital-model/xgboost/
      preprocessor.pkl          (copy từ data_prep output)
      feature_metadata.json     (copy từ data_prep output)
      model_reg.pkl
      model_cls.pkl
      explainer.pkl
  gs://<bucket>/hospital-model/lightgbm/
      (same layout)

Usage:
  python src/train.py \\
      --project_id project-8e2366a6-d3cc-40ee-9de \\
      --bucket_name project-8e2366a6-d3cc-40ee-9de-hospital-model \\
      --artifact_dir /app/ml_artifacts \\
      --xgb_prefix hospital-model/xgboost \\
      --lgb_prefix hospital-model/lightgbm \\
      [--n_estimators 300] \\
      [--max_depth 6] \\
      [--learning_rate 0.05]
"""

import argparse
import io
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from google.cloud import storage
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("train")

# ─── Constants ────────────────────────────────────────────────────────────────

# Hyperparams được đặt theo notebook Advanced_Model_Evaluation
# (XGB: MAE=0.0685, R2=0.8230, ROC-AUC=0.9466 | LGB: MAE=0.0675, R2=0.8297, ROC-AUC=0.9460)
_XGB_DEFAULTS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "n_jobs":           -1,
    "verbosity":        0,
}

_LGB_DEFAULTS = {
    "n_estimators":       300,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_samples":  20,
    "n_jobs":             -1,
    "verbose":            -1,
}

# Threshold phân loại high-strain — khớp với spec §5.6 (>= 90% = high strain)
# Notebook dùng 0.85 cho stress test; production alert flag dùng 0.90.
# Train classifier với 0.85 để có sensitivity cao hơn ở inference.
_STRAIN_THRESHOLD = 0.85


# ─── GCS Helpers ─────────────────────────────────────────────────────────────

def _upload_pkl(gcs_client: storage.Client, bucket_name: str, blob_path: str, obj) -> None:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_file(buf)
    logger.info("Uploaded pkl: gs://%s/%s", bucket_name, blob_path)


def _upload_json(gcs_client: storage.Client, bucket_name: str, blob_path: str, obj: dict) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(data, content_type="application/json")
    logger.info("Uploaded json: gs://%s/%s", bucket_name, blob_path)


def _download_pkl(gcs_client: storage.Client, bucket_name: str, blob_path: str):
    buf = io.BytesIO()
    gcs_client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pickle.load(buf)


def _download_json(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> dict:
    buf = io.BytesIO()
    gcs_client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return json.loads(buf.read().decode("utf-8"))


def _blob_exists(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> bool:
    return gcs_client.bucket(bucket_name).blob(blob_path).exists()

def load_prep_artifacts(
    artifact_dir: str,
    gcs_client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
) -> tuple:
    """
    Load preprocessor.pkl và feature_metadata.json.

    Primary path: local artifact_dir (khi train.py chạy trên same machine với data_prep.py).
    Fallback: GCS (khi train.py chạy trên Vertex AI training job riêng biệt).

    Returns: (preprocessor, feature_metadata)
    """
    local_prep = os.path.join(artifact_dir, "preprocessor.pkl")
    local_meta = os.path.join(artifact_dir, "feature_metadata.json")

    if os.path.exists(local_prep) and os.path.exists(local_meta):
        logger.info("Loading prep artifacts from local: %s", artifact_dir)
        with open(local_prep, "rb") as f:
            preprocessor = pickle.load(f)
        with open(local_meta) as f:
            feature_metadata = json.load(f)
    else:
        logger.info("Local artifacts not found — loading from GCS gs://%s/%s", bucket_name, gcs_prefix)
        prep_blob = f"{gcs_prefix}/preprocessor.pkl"
        meta_blob = f"{gcs_prefix}/feature_metadata.json"
        if not _blob_exists(gcs_client, bucket_name, prep_blob):
            raise FileNotFoundError(
                f"preprocessor.pkl not found at gs://{bucket_name}/{prep_blob}. "
                "Run data_prep.py first."
            )
        preprocessor     = _download_pkl(gcs_client, bucket_name, prep_blob)
        feature_metadata = _download_json(gcs_client, bucket_name, meta_blob)

    logger.info(
        "Prep artifacts loaded: %d features (%d numeric, %d cat)",
        feature_metadata["n_features"],
        len(feature_metadata["numeric_feats"]),
        len(feature_metadata["cat_feats"]),
    )
    return preprocessor, feature_metadata


def load_split_parquets(artifact_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train_features.parquet và val_features.parquet từ artifact_dir.

    data_prep.py saves raw (pre-transform) parquets với cả feature columns lẫn targets.
    train.py đọc lại và gọi preprocessor.transform() để tạo X arrays.
    """
    train_path = os.path.join(artifact_dir, "train_features.parquet")
    val_path   = os.path.join(artifact_dir, "val_features.parquet")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Split parquet not found: {p}. Run data_prep.py first."
            )

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    logger.info(
        "Loaded split parquets: train=%d rows | val=%d rows",
        len(train_df), len(val_df),
    )
    return train_df, val_df


# ─── Feature/Target Extraction ────────────────────────────────────────────────

def extract_Xy(
    df: pd.DataFrame,
    feature_metadata: dict,
    preprocessor,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform feature DataFrame → X array. Extract y_reg và y_cls.

    y_cls = target_high_strain từ feature store (pre-computed binary label).
    Nếu target_high_strain không có trong parquet (schema mismatch), reconstruct
    từ target_occupancy_next_week >= STRAIN_THRESHOLD.
    """
    feature_cols = feature_metadata["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("[%s] %d missing feature cols → NaN: %s", split_name, len(missing), missing)
        for c in missing:
            df[c] = np.nan

    X = preprocessor.transform(df[feature_cols])
    logger.info("[%s] X shape: %s", split_name, X.shape)

    y_reg = df["target_occupancy_next_week"].clip(0, 1).values

    if "target_high_strain" in df.columns:
        y_cls = df["target_high_strain"].values.astype(int)
    else:
        logger.warning(
            "[%s] target_high_strain absent — reconstructing from "
            "target_occupancy_next_week >= %.2f", split_name, _STRAIN_THRESHOLD
        )
        y_cls = (y_reg >= _STRAIN_THRESHOLD).astype(int)

    pos_rate = y_cls.mean()
    logger.info(
        "[%s] target_occupancy: mean=%.3f | target_high_strain: pos_rate=%.1f%%",
        split_name, y_reg.mean(), pos_rate * 100,
    )
    return X, y_reg, y_cls


# ─── Metric Helpers ───────────────────────────────────────────────────────────

def _quick_reg_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(r2_score(y_true, y_pred))
    logger.info("[%s] val regression: MAE=%.4f | RMSE=%.4f | R2=%.4f", name, mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _quick_cls_metrics(name: str, y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    auc  = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    f1   = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec  = float(recall_score(y_true, pred, zero_division=0))
    logger.info(
        "[%s] val classification: ROC-AUC=%.4f | F1=%.4f | Prec=%.4f | Recall=%.4f",
        name, auc, f1, prec, rec,
    )
    return {"roc_auc": auc, "f1": f1, "precision": prec, "recall": rec}


# ─── Training ─────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
    random_seed: int,
    hyperparams: dict,
) -> tuple:
    """
    Train XGBRegressor + XGBClassifier.

    scale_pos_weight cho classifier: tỷ lệ nghịch của class imbalance.
    Fit với eval_set để early stopping nếu cần (hiện disabled để keep n_estimators cố định
    cho reproducibility với notebook).
    """
    logger.info("Training XGBoost Regressor...")
    t0 = time.time()

    xgb_reg = XGBRegressor(
        **hyperparams,
        random_state=random_seed,
    )
    xgb_reg.fit(X_train, y_reg, eval_set=[(X_val, y_val_reg)], verbose=False)

    reg_metrics = _quick_reg_metrics(
        "XGBoost", y_val_reg, xgb_reg.predict(X_val).clip(0, 1)
    )

    logger.info("Training XGBoost Classifier...")
    pos_w = float((y_cls == 0).sum()) / max(float((y_cls == 1).sum()), 1.0)

    xgb_cls = XGBClassifier(
        **{k: v for k, v in hyperparams.items() if k not in ("min_child_weight",)},
        min_child_weight=hyperparams.get("min_child_weight", 5),
        scale_pos_weight=pos_w,
        eval_metric="auc",
        random_state=random_seed,
    )
    xgb_cls.fit(X_train, y_cls, eval_set=[(X_val, y_val_cls)], verbose=False)

    cls_metrics = _quick_cls_metrics(
        "XGBoost", y_val_cls, xgb_cls.predict_proba(X_val)[:, 1]
    )

    logger.info("XGBoost training complete in %.1fs", time.time() - t0)
    return xgb_reg, xgb_cls, {**reg_metrics, **cls_metrics}


def train_lightgbm(
    X_train: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
    random_seed: int,
    hyperparams: dict,
) -> tuple:
    """
    Train LGBMRegressor + LGBMClassifier.

    Early stopping callback (patience=30) giống notebook — LightGBM thường
    converge sớm hơn, early stopping tránh overfit trên val set.
    """
    logger.info("Training LightGBM Regressor...")
    t0 = time.time()

    lgb_reg = LGBMRegressor(**hyperparams, random_state=random_seed)
    lgb_reg.fit(
        X_train, y_reg,
        eval_set=[(X_val, y_val_reg)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    reg_metrics = _quick_reg_metrics(
        "LightGBM", y_val_reg, lgb_reg.predict(X_val).clip(0, 1)
    )

    logger.info("Training LightGBM Classifier...")
    lgb_cls = LGBMClassifier(
        **hyperparams,
        class_weight="balanced",
        random_state=random_seed,
    )
    lgb_cls.fit(
        X_train, y_cls,
        eval_set=[(X_val, y_val_cls)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    cls_metrics = _quick_cls_metrics(
        "LightGBM", y_val_cls, lgb_cls.predict_proba(X_val)[:, 1]
    )

    logger.info("LightGBM training complete in %.1fs", time.time() - t0)
    return lgb_reg, lgb_cls, {**reg_metrics, **cls_metrics}


# ─── SHAP Explainer ───────────────────────────────────────────────────────────

def build_shap_explainer(reg_model, all_feat_names: list[str]) -> shap.TreeExplainer:
    """
    Build TreeExplainer trên regression model.
    SHAP luôn explain reg model (pred_occupancy_next_week) — đây là primary output
    cho Looker Studio và alert engine, không phải classifier.
    """
    logger.info("Building SHAP TreeExplainer (regression model)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(reg_model)
    logger.info("SHAP explainer built in %.1fs", time.time() - t0)
    return explainer


# ─── GCS Upload ───────────────────────────────────────────────────────────────

def upload_model_artifacts(
    gcs_client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
    preprocessor,
    feature_metadata: dict,
    model_reg,
    model_cls,
    explainer: shap.TreeExplainer,
    val_metrics: dict,
    model_tag: str,
    train_timestamp: str,
) -> str:
    """
    Upload artifacts theo GCS layout chuẩn của pipeline.

    Thêm training_info.json để evaluate_and_register.py đọc metrics mà không
    cần re-run training. Đây là bridge giữa train.py và evaluate_and_register.py.
    """
    _upload_pkl(gcs_client, bucket_name, f"{gcs_prefix}/preprocessor.pkl",       preprocessor)
    _upload_pkl(gcs_client, bucket_name, f"{gcs_prefix}/model_reg.pkl",           model_reg)
    _upload_pkl(gcs_client, bucket_name, f"{gcs_prefix}/model_cls.pkl",           model_cls)
    _upload_pkl(gcs_client, bucket_name, f"{gcs_prefix}/explainer.pkl",           explainer)
    _upload_json(gcs_client, bucket_name, f"{gcs_prefix}/feature_metadata.json",  feature_metadata)

    # training_info.json: bridge artifact cho evaluate_and_register.py
    training_info = {
        "model_tag":        model_tag,
        "gcs_prefix":       gcs_prefix,
        "train_timestamp":  train_timestamp,
        "val_metrics":      val_metrics,
        "feature_count":    feature_metadata["n_features"],
        "train_rows":       feature_metadata["split_counts"]["train"],
        "val_rows":         feature_metadata["split_counts"]["val"],
        "train_end":        feature_metadata["train_end"],
        "val_end":          feature_metadata["val_end"],
    }
    _upload_json(gcs_client, bucket_name, f"{gcs_prefix}/training_info.json", training_info)

    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    logger.info("[%s] All artifacts uploaded → %s", model_tag, gcs_uri)
    return gcs_uri


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hospital ML training")
    p.add_argument("--project_id",   required=True)
    p.add_argument("--bucket_name",  required=True,
                   help="GCS bucket cho model artifacts")
    p.add_argument("--artifact_dir", default="/app/ml_artifacts",
                   help="Local directory với data_prep.py outputs (parquets + preprocessor)")
    p.add_argument("--xgb_prefix",   default="hospital-model/xgboost",
                   help="GCS prefix cho XGBoost artifacts")
    p.add_argument("--lgb_prefix",   default="hospital-model/lightgbm",
                   help="GCS prefix cho LightGBM artifacts")
    p.add_argument("--random_seed",  type=int, default=42)
    # Hyperparams — override defaults nếu cần từ Airflow DAG
    p.add_argument("--n_estimators",   type=int,   default=_XGB_DEFAULTS["n_estimators"])
    p.add_argument("--max_depth",      type=int,   default=_XGB_DEFAULTS["max_depth"])
    p.add_argument("--learning_rate",  type=float, default=_XGB_DEFAULTS["learning_rate"])
    p.add_argument("--subsample",      type=float, default=_XGB_DEFAULTS["subsample"])
    p.add_argument("--colsample_bytree", type=float, default=_XGB_DEFAULTS["colsample_bytree"])
    p.add_argument("--min_child_weight", type=int,  default=_XGB_DEFAULTS["min_child_weight"])
    p.add_argument("--min_child_samples", type=int, default=_LGB_DEFAULTS["min_child_samples"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    train_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("=" * 60)
    logger.info("Training START: %s", train_timestamp)
    logger.info("  artifact_dir : %s", args.artifact_dir)
    logger.info("  xgb_prefix   : %s", args.xgb_prefix)
    logger.info("  lgb_prefix   : %s", args.lgb_prefix)
    logger.info("  random_seed  : %d", args.random_seed)
    logger.info("=" * 60)

    gcs_client = storage.Client(project=args.project_id)

    # 1. Load data_prep artifacts
    preprocessor, feature_metadata = load_prep_artifacts(
        args.artifact_dir, gcs_client, args.bucket_name, args.xgb_prefix
    )
    all_feat_names = feature_metadata["feature_cols"]

    # 2. Load split parquets
    train_df, val_df = load_split_parquets(args.artifact_dir)

    # 3. Extract X, y cho mỗi split
    X_train, y_tr_reg, y_tr_cls = extract_Xy(train_df, feature_metadata, preprocessor, "train")
    X_val,   y_va_reg, y_va_cls = extract_Xy(val_df,   feature_metadata, preprocessor, "val")

    # 4. Build hyperparams từ args
    xgb_hp = {
        "n_estimators":     args.n_estimators,
        "max_depth":        args.max_depth,
        "learning_rate":    args.learning_rate,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    lgb_hp = {
        "n_estimators":      args.n_estimators,
        "max_depth":         args.max_depth,
        "learning_rate":     args.learning_rate,
        "subsample":         args.subsample,
        "colsample_bytree":  args.colsample_bytree,
        "min_child_samples": args.min_child_samples,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    # 5. Train XGBoost
    xgb_reg, xgb_cls, xgb_val_metrics = train_xgboost(
        X_train, y_tr_reg, y_tr_cls,
        X_val,   y_va_reg, y_va_cls,
        args.random_seed, xgb_hp,
    )
    xgb_explainer = build_shap_explainer(xgb_reg, all_feat_names)

    # 6. Train LightGBM
    lgb_reg, lgb_cls, lgb_val_metrics = train_lightgbm(
        X_train, y_tr_reg, y_tr_cls,
        X_val,   y_va_reg, y_va_cls,
        args.random_seed, lgb_hp,
    )
    lgb_explainer = build_shap_explainer(lgb_reg, all_feat_names)

    # 7. Upload XGBoost artifacts
    upload_model_artifacts(
        gcs_client=gcs_client,
        bucket_name=args.bucket_name,
        gcs_prefix=args.xgb_prefix,
        preprocessor=preprocessor,
        feature_metadata=feature_metadata,
        model_reg=xgb_reg,
        model_cls=xgb_cls,
        explainer=xgb_explainer,
        val_metrics=xgb_val_metrics,
        model_tag="XGBoost",
        train_timestamp=train_timestamp,
    )

    # 8. Upload LightGBM artifacts
    upload_model_artifacts(
        gcs_client=gcs_client,
        bucket_name=args.bucket_name,
        gcs_prefix=args.lgb_prefix,
        preprocessor=preprocessor,
        feature_metadata=feature_metadata,
        model_reg=lgb_reg,
        model_cls=lgb_cls,
        explainer=lgb_explainer,
        val_metrics=lgb_val_metrics,
        model_tag="LightGBM",
        train_timestamp=train_timestamp,
    )

    logger.info("=" * 60)
    logger.info("Training COMPLETE")
    logger.info("  XGBoost  val MAE=%.4f | R2=%.4f | ROC-AUC=%.4f",
                xgb_val_metrics["mae"], xgb_val_metrics["r2"], xgb_val_metrics["roc_auc"])
    logger.info("  LightGBM val MAE=%.4f | R2=%.4f | ROC-AUC=%.4f",
                lgb_val_metrics["mae"], lgb_val_metrics["r2"], lgb_val_metrics["roc_auc"])
    logger.info("  Artifacts → gs://%s/{%s,%s}", args.bucket_name, args.xgb_prefix, args.lgb_prefix)
    logger.info("  Next step: run evaluate_and_register.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()