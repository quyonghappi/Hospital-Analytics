"""
evaluate_and_register.py — Evaluation gate + Vertex AI Model Registry.

Đọc training_info.json (bridge từ train.py), chạy test-set evaluation độc lập,
áp gate logic, và đăng ký model artifacts vào Vertex AI Model Registry.

Vị trí trong Airflow DAG:
  data_prep.py >> train.py >> evaluate_and_register.py >> inference.py >> post_process.py

Gate logic (ALL conditions phải pass):
  1. test MAE  < mae_threshold      (default 0.10  — 10% absolute error trên range 0-1)
  2. test R2   > r2_threshold       (default 0.75)
  3. test AUC  > roc_auc_threshold  (default 0.85)
  4. [if incumbent exists] new_MAE ≤ incumbent_MAE + incumbent_tolerance (default 0.005)

  GATE FAIL → sys.exit(2): Airflow marks task FAILED, downstream inference không chạy.

Registration:
  Cả XGBoost và LightGBM được evaluate và register nếu pass gate.
  Champion (test MAE thấp hơn) → labels.status = production.
  Runner-up → labels.status = candidate.
  Tie-break: XGBoost được ưu tiên (primary serving model theo architecture decision).
  Nếu chỉ 1 model pass gate → model đó = production.

Vertex AI label conventions (queryable từ Airflow / inference orchestration):
  status:              production | candidate
  model_type:          xgboost | lightgbm
  train_date:          YYYYMMDD
  test_mae_millimae:   int(mae * 1000)  — integer để sortable trong label value
  test_r2_centesimal:  int(r2  * 100)
  test_auc_centesimal: int(auc * 100), hoặc "0" nếu single-class test set

NOTE về serving container:
  Pipeline dùng inference.py (Vertex AI Custom Job) — không dùng Vertex AI Managed
  Online/Batch Prediction. serving_container_image_uri được pass vào Model.upload()
  cho artifact registration / governance, không phải live serving.

Usage:
  python src/evaluate_and_register.py \\
      --project_id project-8e2366a6-d3cc-40ee-9de \\
      --region asia-southeast1 \\
      --bucket_name <BUCKET_NAME> \\
      --artifact_dir /app/ml_artifacts \\
      [--xgb_prefix hospital-model/xgboost] \\
      [--lgb_prefix hospital-model/lightgbm] \\
      [--mae_threshold 0.10] \\
      [--r2_threshold 0.75] \\
      [--roc_auc_threshold 0.85] \\
      [--incumbent_tolerance 0.005] \\
      [--skip_lgb] \\
      [--dry_run]
"""

import argparse
import io
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import aiplatform, storage
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate_and_register")

_DEFAULT_MAE_THRESHOLD       = 0.10    # 10% absolute error — calibrated với notebook baseline
_DEFAULT_R2_THRESHOLD        = 0.75    # notebook achieved 0.82-0.83; threshold cho phép regression
_DEFAULT_ROC_AUC_THRESHOLD   = 0.85    # notebook achieved 0.94-0.95
_DEFAULT_INCUMBENT_TOLERANCE = 0.005   # 0.5% MAE regression tolerance — buffer cho distribution shift

# Serving containers — dùng cho artifact registration, không phải managed serving
_CONTAINER_XGB = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
_CONTAINER_LGB = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"

_DISPLAY_NAMES = {
    "XGBoost":  "hospital-occupancy-xgboost",
    "LightGBM": "hospital-occupancy-lightgbm",
}

# Strain threshold cho reconstruction nếu target_high_strain absent từ parquet
# Phải match với train.py _STRAIN_THRESHOLD
_STRAIN_THRESHOLD = 0.85

def _load_pkl(gcs_client: storage.Client, bucket_name: str, blob_path: str):
    buf = io.BytesIO()
    gcs_client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pickle.load(buf)


def _load_json(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> dict:
    buf = io.BytesIO()
    gcs_client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return json.loads(buf.read().decode("utf-8"))


def _upload_json(
    gcs_client: storage.Client,
    bucket_name: str,
    blob_path: str,
    obj: dict,
) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(
        data, content_type="application/json"
    )
    logger.info("Uploaded: gs://%s/%s", bucket_name, blob_path)


def _blob_exists(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> bool:
    return gcs_client.bucket(bucket_name).blob(blob_path).exists()


def load_training_info(
    gcs_client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
    model_tag: str,
) -> dict:
    """
    Load training_info.json từ GCS — bridge artifact được export bởi train.py.

    training_info.json chứa: val_metrics, feature_count, train/val rows,
    train_end/val_end, gcs_prefix, train_timestamp.

    Fail fast nếu file không tồn tại — không nên evaluate model chưa được train.
    """
    blob_path = f"{gcs_prefix}/training_info.json"
    if not _blob_exists(gcs_client, bucket_name, blob_path):
        raise FileNotFoundError(
            f"[{model_tag}] training_info.json không tìm thấy: "
            f"gs://{bucket_name}/{blob_path}. Run train.py trước."
        )
    info = _load_json(gcs_client, bucket_name, blob_path)
    logger.info(
        "[%s] training_info: val_MAE=%.4f | val_R2=%.4f | val_AUC=%.4f | "
        "features=%d | train_rows=%d | val_rows=%d",
        model_tag,
        info["val_metrics"].get("mae",     float("nan")),
        info["val_metrics"].get("r2",      float("nan")),
        info["val_metrics"].get("roc_auc", float("nan")),
        info.get("feature_count", -1),
        info.get("train_rows",    -1),
        info.get("val_rows",      -1),
    )
    return info

def load_test_artifacts(
    artifact_dir: str,
    gcs_client: storage.Client,
    bucket_name: str,
    reference_prefix: str,
) -> tuple[pd.DataFrame, object, dict]:
    """
    Load test_features.parquet + preprocessor.pkl + feature_metadata.json.

    Priority: local artifact_dir → GCS fallback (khi chạy trên Vertex AI Job riêng).

    NOTE: test_features.parquet CHỈ tồn tại trong local artifact_dir.
    data_prep.py không upload test parquet lên GCS theo thiết kế (tránh test data leak).
    Khi chạy trên Vertex AI, artifact_dir phải được mount từ GCS bucket riêng
    hoặc truyền qua trong cùng training job với data_prep.py.

    reference_prefix: GCS prefix để fallback load preprocessor + metadata
    (thường là xgb_prefix vì preprocessor shared giữa 2 models).
    """
    test_path = os.path.join(artifact_dir, "test_features.parquet")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"test_features.parquet không tìm thấy: {test_path}. "
            "Đảm bảo data_prep.py đã chạy và artifact_dir được mount đúng."
        )
    test_df = pd.read_parquet(test_path)
    logger.info(
        "[QA] Test set: %d rows × %d cols | hospitals=%d | date_range=%s → %s",
        len(test_df),
        test_df.shape[1],
        test_df["hospital_id"].nunique() if "hospital_id" in test_df.columns else -1,
        str(test_df["report_date"].min())[:10] if "report_date" in test_df.columns else "?",
        str(test_df["report_date"].max())[:10] if "report_date" in test_df.columns else "?",
    )

    local_prep = os.path.join(artifact_dir, "preprocessor.pkl")
    local_meta = os.path.join(artifact_dir, "feature_metadata.json")

    if os.path.exists(local_prep) and os.path.exists(local_meta):
        with open(local_prep, "rb") as f:
            preprocessor = pickle.load(f)
        with open(local_meta) as f:
            feature_metadata = json.load(f)
        logger.info("Prep artifacts: local (%s)", artifact_dir)
    else:
        logger.info("Local prep artifacts không tìm thấy → fallback GCS prefix: %s", reference_prefix)
        preprocessor     = _load_pkl(gcs_client, bucket_name, f"{reference_prefix}/preprocessor.pkl")
        feature_metadata = _load_json(gcs_client, bucket_name, f"{reference_prefix}/feature_metadata.json")

    logger.info(
        "feature_metadata: %d features (%d numeric, %d cat) | train_end=%s | val_end=%s",
        feature_metadata["n_features"],
        len(feature_metadata["numeric_feats"]),
        len(feature_metadata["cat_feats"]),
        feature_metadata["train_end"],
        feature_metadata["val_end"],
    )
    return test_df, preprocessor, feature_metadata

def evaluate_on_test(
    test_df: pd.DataFrame,
    preprocessor,
    feature_metadata: dict,
    gcs_client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
    model_tag: str,
) -> dict:
    """
    Full test-set evaluation: transform → predict (reg + cls) → metric suite.

    Model artifacts được load từ GCS — đảm bảo evaluate đúng artifacts sẽ
    được registered. Không evaluate in-memory objects để tránh mismatch.

    Regression:     MAE, RMSE, R2, MAPE
    Classification: ROC-AUC, F1, Precision, Recall
    Distribution:   avg_pred_occ, alert_rate (pred >= 0.90), prediction_bias

    MAPE: denominator được clip ở 0.01 để tránh div/0 khi occupancy gần 0
    (edge case: hospitals mới mở hoặc được đóng cửa tạm thời).
    """
    feature_cols = feature_metadata["feature_cols"]

    # Schema alignment guard — backfill missing columns với NaN
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        logger.warning(
            "[%s] %d features absent từ test set → NaN backfill: %s",
            model_tag, len(missing), missing,
        )
        for c in missing:
            test_df[c] = np.nan
    else:
        logger.info("[%s] Column alignment: %d features present ✓", model_tag, len(feature_cols))

    # Transform
    X_test = preprocessor.transform(test_df[feature_cols])
    logger.info("[%s] X_test shape: %s", model_tag, X_test.shape)

    # Load canonical model artifacts từ GCS
    logger.info("[%s] Loading model artifacts từ GCS...", model_tag)
    reg = _load_pkl(gcs_client, bucket_name, f"{gcs_prefix}/model_reg.pkl")
    cls = _load_pkl(gcs_client, bucket_name, f"{gcs_prefix}/model_cls.pkl")

    # ── Regression ────────────────────────────────────────────────────────────
    y_true_reg = test_df["target_occupancy_next_week"].clip(0, 1).values
    y_pred_reg = reg.predict(X_test).clip(0, 1)

    mae  = float(mean_absolute_error(y_true_reg, y_pred_reg))
    rmse = float(np.sqrt(np.mean((y_true_reg - y_pred_reg) ** 2)))
    r2   = float(r2_score(y_true_reg, y_pred_reg))
    mape = float(
        np.mean(np.abs((y_true_reg - y_pred_reg) / np.clip(y_true_reg, 0.01, None))) * 100
    )
    pred_bias = float((y_pred_reg - y_true_reg).mean())  # positive = over-predict

    logger.info(
        "[%s] Regression — MAE=%.4f | RMSE=%.4f | R2=%.4f | MAPE=%.2f%% | bias=%.4f",
        model_tag, mae, rmse, r2, mape, pred_bias,
    )

    # ── Classification ────────────────────────────────────────────────────────
    if "target_high_strain" in test_df.columns:
        y_true_cls = test_df["target_high_strain"].values.astype(int)
    else:
        logger.warning(
            "[%s] target_high_strain absent — reconstruct từ occ >= %.2f",
            model_tag, _STRAIN_THRESHOLD,
        )
        y_true_cls = (y_true_reg >= _STRAIN_THRESHOLD).astype(int)

    y_prob_cls = cls.predict_proba(X_test)[:, 1]
    y_pred_cls = (y_prob_cls >= 0.5).astype(int)

    # ROC-AUC không tính được nếu test set chỉ có 1 class (valid nếu test period ngắn)
    has_both_classes = len(np.unique(y_true_cls)) > 1
    if not has_both_classes:
        logger.warning(
            "[%s] Test set chỉ có 1 class trong target_high_strain "
            "(pos_rate=%.1f%%) — ROC-AUC = NaN.",
            model_tag, y_true_cls.mean() * 100,
        )
    auc  = float(roc_auc_score(y_true_cls, y_prob_cls)) if has_both_classes else float("nan")
    f1   = float(f1_score(y_true_cls, y_pred_cls, zero_division=0))
    prec = float(precision_score(y_true_cls, y_pred_cls, zero_division=0))
    rec  = float(recall_score(y_true_cls, y_pred_cls, zero_division=0))

    logger.info(
        "[%s] Classification — ROC-AUC=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f",
        model_tag, auc, f1, prec, rec,
    )

    # ── Distribution QA ───────────────────────────────────────────────────────
    n_total    = len(y_pred_reg)
    n_alerts   = int((y_pred_reg >= 0.90).sum())
    alert_rate = n_alerts / n_total if n_total > 0 else 0.0

    logger.info(
        "[%s] Distribution — avg_pred=%.1f%% | avg_actual=%.1f%% | "
        "alerts(>=90%%)=%d/%d (%.1f%%)",
        model_tag,
        y_pred_reg.mean() * 100,
        y_true_reg.mean() * 100,
        n_alerts, n_total, alert_rate * 100,
    )

    return {
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
        "mape":         mape,
        "pred_bias":    pred_bias,
        "roc_auc":      auc,
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "n_test_rows":  n_total,
        "alert_rate":   alert_rate,
    }


# ─── Gate Logic ──────────────────────────────────────────────────────────────

def apply_gate(
    model_tag: str,
    test_metrics: dict,
    incumbent_test_mae: Optional[float],
    mae_threshold: float,
    r2_threshold: float,
    roc_auc_threshold: float,
    incumbent_tolerance: float,
) -> tuple[bool, list[str]]:
    """
    Evaluate gate. Returns (passed: bool, failures: list[str]).

    Condition 4 (incumbent comparison): cho phép MAE tệ hơn incumbent tối đa
    incumbent_tolerance (default 0.005 = 0.5%). Buffer này cần thiết vì:
      - Distribution shift ngắn hạn có thể làm model mới metrics tệ hơn tạm thời
      - Nếu threshold quá strict → block valid retraining, model cũ sẽ stale

    Nếu test set chỉ có 1 class (ROC-AUC = NaN) → skip condition 3.
    """
    failures: list[str] = []

    if test_metrics["mae"] >= mae_threshold:
        failures.append(
            f"MAE={test_metrics['mae']:.4f} ≥ threshold={mae_threshold:.4f}"
        )

    if test_metrics["r2"] <= r2_threshold:
        failures.append(
            f"R2={test_metrics['r2']:.4f} ≤ threshold={r2_threshold:.4f}"
        )

    if (not np.isnan(test_metrics["roc_auc"])
            and test_metrics["roc_auc"] <= roc_auc_threshold):
        failures.append(
            f"ROC-AUC={test_metrics['roc_auc']:.4f} ≤ threshold={roc_auc_threshold:.4f}"
        )

    if incumbent_test_mae is not None:
        ceiling = incumbent_test_mae + incumbent_tolerance
        if test_metrics["mae"] > ceiling:
            failures.append(
                f"MAE={test_metrics['mae']:.4f} > "
                f"incumbent_MAE={incumbent_test_mae:.4f} + tol={incumbent_tolerance:.4f} "
                f"= {ceiling:.4f}"
            )
        else:
            delta = incumbent_test_mae - test_metrics["mae"]
            direction = f"+{delta:.4f} improvement" if delta >= 0 else f"{delta:.4f} (regression within tolerance)"
            logger.info(
                "[GATE][%s] Incumbent comparison: %s ✓",
                model_tag, direction,
            )

    passed = len(failures) == 0
    logger.info("[GATE][%s] %s", model_tag, "PASS ✓" if passed else f"FAIL — {len(failures)} violation(s)")
    for reason in failures:
        logger.warning("[GATE][%s] %s", model_tag, reason)

    return passed, failures


# ─── Incumbent Query ─────────────────────────────────────────────────────────

def get_incumbent_info(display_name: str) -> tuple[Optional[str], Optional[float]]:
    """
    Query Vertex AI Model Registry cho production model hiện tại của display_name.

    Returns: (resource_name, test_mae) hoặc (None, None) nếu chưa có production model.

    Incumbent được xác định bằng label status=production. Lấy model mới nhất
    (order_by create_time desc) trong trường hợp có nhiều production versions.

    Error handling: nếu Vertex AI query thất bại (permissions, quota, network) →
    log warning và skip incumbent comparison, không block pipeline.
    Labels.test_mae_millimae là int để workaround label value type limitations.
    """
    try:
        models = aiplatform.Model.list(
            filter=f'display_name="{display_name}" AND labels.status="production"',
            order_by="create_time desc",
        )
    except Exception as e:
        logger.warning(
            "[Incumbent] Query Vertex AI thất bại cho '%s': %s. "
            "Bỏ qua incumbent comparison — chỉ áp absolute thresholds.",
            display_name, e,
        )
        return None, None

    if not models:
        logger.info(
            "[Incumbent] Không tìm thấy production model '%s' — first-time registration.",
            display_name,
        )
        return None, None

    incumbent = models[0]
    resource_name = incumbent.resource_name

    millimae_str = (incumbent.labels or {}).get("test_mae_millimae")
    if millimae_str:
        test_mae = int(millimae_str) / 1000.0
        logger.info(
            "[Incumbent] Found: %s | test_MAE=%.4f",
            resource_name, test_mae,
        )
    else:
        test_mae = None
        logger.warning(
            "[Incumbent] %s: test_mae_millimae label missing. "
            "Bỏ qua relative gate check — chỉ áp absolute thresholds.",
            resource_name,
        )

    return resource_name, test_mae


# ─── Model Registration ──────────────────────────────────────────────────────

def register_model(
    display_name: str,
    artifact_uri: str,
    serving_container_image_uri: str,
    model_tag: str,
    train_info: dict,
    test_metrics: dict,
    status: str,
    incumbent_resource_name: Optional[str],
    dry_run: bool,
) -> Optional[str]:
    """
    Register model artifacts vào Vertex AI Model Registry.

    Nếu incumbent_resource_name có → tạo new version (parent_model argument).
    Nếu không có → tạo model entry mới.

    Tất cả label values phải là strings (Vertex AI requirement).
    Numeric metrics được encode thành integers × multiplier để sortable trong labels.

    Returns: resource_name nếu registered. None nếu dry_run.
    """
    mae = test_metrics["mae"]
    r2  = test_metrics["r2"]
    auc = test_metrics["roc_auc"]

    labels = {
        "status":               status,
        "model_type":           model_tag.lower(),
        # Dates: YYYYMMDD (không dùng dấu '-' — label format constraint)
        "train_date":           train_info["train_timestamp"][:10].replace("-", ""),
        "train_end_date":       train_info["train_end"].replace("-", ""),
        # Metrics as integers để sortable khi query
        "test_mae_millimae":    str(int(mae * 1000)),
        "test_r2_centesimal":   str(max(0, int(r2 * 100))),
        "test_auc_centesimal":  str(int(auc * 100)) if not np.isnan(auc) else "0",
        # Training info
        "feature_count":        str(train_info.get("feature_count", -1)),
        "train_rows":           str(train_info.get("train_rows", -1)),
    }

    description = (
        f"{model_tag} | train_end={train_info['train_end']} | "
        f"test MAE={mae:.4f} R2={r2:.4f} AUC={auc:.4f} | "
        f"features={train_info.get('feature_count', '?')} | status={status}"
    )

    logger.info(
        "[%s] Registering → display_name=%s | uri=%s | status=%s",
        model_tag, display_name, artifact_uri, status,
    )
    logger.info("[%s] Labels: %s", model_tag, labels)

    if dry_run:
        logger.info("[%s] DRY RUN — registration skipped.", model_tag)
        return None

    upload_kwargs: dict = {
        "display_name":                display_name,
        "artifact_uri":                artifact_uri,
        "serving_container_image_uri": serving_container_image_uri,
        "description":                 description,
        "labels":                      labels,
        "sync":                        True,
    }

    if incumbent_resource_name:
        # Tạo new version của model hiện có — giữ version history
        upload_kwargs["parent_model"] = incumbent_resource_name
        logger.info("[%s] → new version of incumbent: %s", model_tag, incumbent_resource_name)
    else:
        logger.info("[%s] → new model entry (no incumbent found)", model_tag)

    model = aiplatform.Model.upload(**upload_kwargs)
    logger.info("[%s] Registered ✓: %s", model_tag, model.resource_name)
    return model.resource_name


# ─── Evaluation Report ────────────────────────────────────────────────────────

def build_evaluation_report(
    model_tag: str,
    train_info: dict,
    val_metrics: dict,
    test_metrics: dict,
    gate_passed: bool,
    gate_failures: list[str],
    registered_status: Optional[str],
    vertex_resource_name: Optional[str],
    evaluation_timestamp: str,
) -> dict:
    """
    Structured evaluation report.

    Được lưu vào:
      - GCS: gs://<bucket>/<prefix>/evaluation_report.json  (audit trail)
      - Local: <artifact_dir>/evaluation_report_<model>.json  (Airflow XCom access)

    Downstream Airflow tasks có thể đọc vertex_resource_name từ report
    để resolve production model GCS prefix cho inference.py.
    """
    return {
        "model_tag":            model_tag,
        "evaluation_timestamp": evaluation_timestamp,
        "gate_passed":          gate_passed,
        "gate_failures":        gate_failures,
        "registered_status":    registered_status,
        "vertex_resource_name": vertex_resource_name,
        "train_summary": {
            "gcs_prefix":      train_info.get("gcs_prefix"),
            "train_end":       train_info.get("train_end"),
            "val_end":         train_info.get("val_end"),
            "train_rows":      train_info.get("train_rows"),
            "val_rows":        train_info.get("val_rows"),
            "feature_count":   train_info.get("feature_count"),
            "train_timestamp": train_info.get("train_timestamp"),
        },
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hospital ML evaluation gate + Vertex AI Model Registry"
    )
    p.add_argument("--project_id",   required=True)
    p.add_argument("--region",       default="asia-southeast1",
                   help="GCP region cho Vertex AI Model Registry")
    p.add_argument("--bucket_name",  required=True,
                   help="GCS bucket chứa model artifacts từ train.py")
    p.add_argument("--artifact_dir", default="/app/ml_artifacts",
                   help="Local dir với test_features.parquet + preprocessor.pkl (từ data_prep.py)")
    p.add_argument("--xgb_prefix",   default="hospital-model/xgboost",
                   help="GCS prefix cho XGBoost artifacts")
    p.add_argument("--lgb_prefix",   default="hospital-model/lightgbm",
                   help="GCS prefix cho LightGBM artifacts")
    # Gate thresholds
    p.add_argument("--mae_threshold",       type=float, default=_DEFAULT_MAE_THRESHOLD,
                   help=f"Max test MAE (default {_DEFAULT_MAE_THRESHOLD})")
    p.add_argument("--r2_threshold",        type=float, default=_DEFAULT_R2_THRESHOLD,
                   help=f"Min test R2 (default {_DEFAULT_R2_THRESHOLD})")
    p.add_argument("--roc_auc_threshold",   type=float, default=_DEFAULT_ROC_AUC_THRESHOLD,
                   help=f"Min test ROC-AUC (default {_DEFAULT_ROC_AUC_THRESHOLD})")
    p.add_argument("--incumbent_tolerance", type=float, default=_DEFAULT_INCUMBENT_TOLERANCE,
                   help=f"Max MAE regression vs incumbent (default {_DEFAULT_INCUMBENT_TOLERANCE})")
    # Options
    p.add_argument("--dry_run",  action="store_true",
                   help="Evaluate và log kết quả nhưng KHÔNG gọi Vertex AI registration")
    p.add_argument("--skip_lgb", action="store_true",
                   help="Bỏ qua LightGBM (khi lgb_prefix artifacts chưa available)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    evaluation_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("=" * 60)
    logger.info("Evaluate & Register START: %s", evaluation_timestamp)
    logger.info("  project_id        : %s", args.project_id)
    logger.info("  region            : %s", args.region)
    logger.info("  artifact_dir      : %s", args.artifact_dir)
    logger.info("  xgb_prefix        : %s", args.xgb_prefix)
    logger.info("  lgb_prefix        : %s", args.lgb_prefix)
    logger.info("  mae_threshold     : %.4f", args.mae_threshold)
    logger.info("  r2_threshold      : %.4f", args.r2_threshold)
    logger.info("  roc_auc_threshold : %.4f", args.roc_auc_threshold)
    logger.info("  incumbent_tol     : %.4f", args.incumbent_tolerance)
    logger.info("  dry_run           : %s",   args.dry_run)
    logger.info("  skip_lgb          : %s",   args.skip_lgb)
    logger.info("=" * 60)

    # Init GCP clients
    gcs_client = storage.Client(project=args.project_id)
    aiplatform.init(project=args.project_id, location=args.region)

    # Build model config list
    model_configs: list[tuple[str, str, str]] = [
        ("XGBoost", args.xgb_prefix, _CONTAINER_XGB),
    ]
    if not args.skip_lgb:
        model_configs.append(("LightGBM", args.lgb_prefix, _CONTAINER_LGB))

    # Load shared test artifacts (preprocessor và feature_metadata shared giữa 2 models)
    # Dùng xgb_prefix làm reference prefix cho fallback load từ GCS
    test_df, preprocessor, feature_metadata = load_test_artifacts(
        args.artifact_dir,
        gcs_client,
        args.bucket_name,
        reference_prefix=args.xgb_prefix,
    )

    # ─── Phase 1: Evaluate all models ─────────────────────────────────────────
    model_results: dict[str, dict] = {}

    for model_tag, gcs_prefix, container_uri in model_configs:
        logger.info("─" * 40)
        logger.info("[Phase 1] Evaluating: %s", model_tag)

        try:
            train_info = load_training_info(
                gcs_client, args.bucket_name, gcs_prefix, model_tag
            )
        except FileNotFoundError as e:
            if model_tag == "XGBoost":
                # XGBoost là primary model — không thể proceed nếu thiếu
                raise
            logger.warning(
                "[%s] training_info.json không tìm thấy — bỏ qua: %s", model_tag, e
            )
            continue

        test_metrics = evaluate_on_test(
            test_df.copy(),  # copy để tránh mutation cross-model
            preprocessor,
            feature_metadata,
            gcs_client,
            args.bucket_name,
            gcs_prefix,
            model_tag,
        )

        display_name = _DISPLAY_NAMES[model_tag]
        incumbent_resource_name, incumbent_test_mae = get_incumbent_info(display_name)

        gate_passed, gate_failures = apply_gate(
            model_tag=model_tag,
            test_metrics=test_metrics,
            incumbent_test_mae=incumbent_test_mae,
            mae_threshold=args.mae_threshold,
            r2_threshold=args.r2_threshold,
            roc_auc_threshold=args.roc_auc_threshold,
            incumbent_tolerance=args.incumbent_tolerance,
        )

        model_results[model_tag] = {
            "gcs_prefix":              gcs_prefix,
            "container_uri":           container_uri,
            "display_name":            display_name,
            "train_info":              train_info,
            "val_metrics":             train_info["val_metrics"],
            "test_metrics":            test_metrics,
            "gate_passed":             gate_passed,
            "gate_failures":           gate_failures,
            "incumbent_resource_name": incumbent_resource_name,
        }

    # ─── Phase 2: Gate summary + abort check ─────────────────────────────────
    logger.info("─" * 40)
    logger.info("[Phase 2] Gate summary:")

    passing_models = {t: r for t, r in model_results.items() if r["gate_passed"]}
    failing_models = {t: r for t, r in model_results.items() if not r["gate_passed"]}

    for tag, result in failing_models.items():
        logger.error("[Gate FAIL] %s:", tag)
        for reason in result["gate_failures"]:
            logger.error("    → %s", reason)

    if not passing_models:
        logger.error("=" * 60)
        logger.error("ALL MODELS FAILED GATE — inference pipeline sẽ không chạy.")
        logger.error("Possible causes:")
        logger.error("  - Data quality regression trong feature store")
        logger.error("  - Feature schema drift (new columns, renamed features)")
        logger.error("  - Temporal distribution shift (train/test gap quá lớn)")
        logger.error("  - train_end/val_end dates cần review (data_prep.py args)")
        logger.error("Action: review evaluation_report JSON, recheck data_prep split dates.")
        logger.error("=" * 60)
        sys.exit(2)  # Airflow marks task FAILED — downstream tasks không chạy

    # ─── Phase 3: Xác định champion ───────────────────────────────────────────
    # Champion = model có test MAE thấp nhất trong passing models
    # Tie-break: XGBoost được ưu tiên (primary serving model theo pipeline architecture)
    champion_tag = min(
        passing_models,
        key=lambda t: (
            passing_models[t]["test_metrics"]["mae"],
            0 if t == "XGBoost" else 1,
        ),
    )
    logger.info(
        "[Phase 3] Champion: %s (test_MAE=%.4f)",
        champion_tag,
        passing_models[champion_tag]["test_metrics"]["mae"],
    )
    if len(passing_models) > 1:
        for tag, result in passing_models.items():
            if tag != champion_tag:
                logger.info(
                    "[Phase 3] Runner-up: %s (test_MAE=%.4f) → status=candidate",
                    tag, result["test_metrics"]["mae"],
                )

    # ─── Phase 4: Register all passing models ─────────────────────────────────
    logger.info("─" * 40)
    logger.info("[Phase 4] Registering %d model(s)...", len(passing_models))

    evaluation_reports: dict[str, dict] = {}

    for model_tag, result in model_results.items():
        if not result["gate_passed"]:
            registered_status    = None
            vertex_resource_name = None
        else:
            registered_status = "production" if model_tag == champion_tag else "candidate"

            vertex_resource_name = register_model(
                display_name=result["display_name"],
                artifact_uri=f"gs://{args.bucket_name}/{result['gcs_prefix']}",
                serving_container_image_uri=result["container_uri"],
                model_tag=model_tag,
                train_info=result["train_info"],
                test_metrics=result["test_metrics"],
                status=registered_status,
                incumbent_resource_name=result["incumbent_resource_name"],
                dry_run=args.dry_run,
            )

        # Build report
        report = build_evaluation_report(
            model_tag=model_tag,
            train_info=result["train_info"],
            val_metrics=result["val_metrics"],
            test_metrics=result["test_metrics"],
            gate_passed=result["gate_passed"],
            gate_failures=result["gate_failures"],
            registered_status=registered_status,
            vertex_resource_name=vertex_resource_name,
            evaluation_timestamp=evaluation_timestamp,
        )
        evaluation_reports[model_tag] = report

        # Upload report lên GCS — audit trail + accessible cho downstream tooling
        _upload_json(
            gcs_client,
            args.bucket_name,
            f"{result['gcs_prefix']}/evaluation_report.json",
            report,
        )

        # Save locally cho Airflow XCom access (Airflow task có thể đọc file này
        # để lấy vertex_resource_name của production model cho inference job)
        local_report_path = os.path.join(
            args.artifact_dir,
            f"evaluation_report_{model_tag.lower()}.json",
        )
        os.makedirs(args.artifact_dir, exist_ok=True)
        with open(local_report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("[%s] Report saved: %s", model_tag, local_report_path)

    # ─── Final Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluate & Register COMPLETE: %s", evaluation_timestamp)
    logger.info("")
    logger.info("  %-10s  %-4s  %-8s  %-6s  %-6s  %-11s  %s",
                "Model", "Gate", "MAE", "R2", "AUC", "Status", "Vertex Resource")
    logger.info("  " + "-" * 78)
    for tag, report in evaluation_reports.items():
        m = report["test_metrics"]
        logger.info(
            "  %-10s  %-4s  %.4f    %.4f  %.4f  %-11s  %s",
            tag,
            "PASS" if report["gate_passed"] else "FAIL",
            m["mae"], m["r2"], m["roc_auc"],
            report.get("registered_status") or "not_registered",
            report.get("vertex_resource_name") or "N/A",
        )
    logger.info("")
    logger.info("  Champion : %s", champion_tag)
    logger.info(
        "  Next     : python src/inference.py --xgb_prefix=%s ...",
        model_results.get("XGBoost", {}).get("gcs_prefix", args.xgb_prefix),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()