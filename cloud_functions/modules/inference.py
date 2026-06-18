import io
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage

from .xai_shap import compute_shap_values, extract_top_n_features, get_or_build_explainer

logger = logging.getLogger(__name__)

_EXCLUDE_COLS = frozenset({
    "hospital_id", "report_date", "county_fips", "zip_code", "county_name",
    "hospital_name", "city", "feature_computed_at",
    "target_occupancy_next_week", "target_high_strain", "occupancy_rate",
})

_CATEGORICAL_COLS = frozenset({
    "state", "hospital_type", "season", "disease_season",
    "healthcare_risk_level", "metro_nonmetro_flag", "hrr_region",
})

_SENTINEL_VALUE = -9999
_SENTINEL_COLS = [
    "icu_used", "icu_occupancy_rate", "covid_patients",
    "covid_icu", "flu_patients", "flu_icu", "covid_admit_adult",
]

_RAW_SCHEMA = [
    bigquery.SchemaField("model_name",               "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("hospital_id",              "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("report_date",              "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("run_date",                 "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("pred_occupancy_next_week", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("pred_high_strain",         "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("pred_high_strain_prob",    "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("actual_occupancy",         "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("actual_high_strain",       "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("abs_error",                "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("shap_base_value",          "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top1_feature",             "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top1_shap",                "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top1_direction",           "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top2_feature",             "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top2_shap",                "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top2_direction",           "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top3_feature",             "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top3_shap",                "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top3_direction",           "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("explanation",              "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("model_version",            "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("data_source",              "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("predicted_at",             "TIMESTAMP", mode="REQUIRED"),
]


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


def _exists(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> bool:
    return gcs_client.bucket(bucket_name).blob(blob_path).exists()


def _get_blob_version(
    gcs_client: storage.Client, bucket_name: str, blob_path: str
) -> str:
    """
    Derive model_version from GCS blob generation number.
    Generation is a stable, unique identifier per object overwrite.
    Format: "<prefix>@gen-<generation>"
    """
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    blob.reload()
    return f"{blob_path}@gen-{blob.generation}"

def load_feature_data(
    bq_client: bigquery.Client,
    source_table: str,
    run_date: str,
    lookback_days: int,
) -> pd.DataFrame:
    logger.info(
        "Loading latest feature snapshot per hospital from %s "
        "(run_date=%s, lookback=%d days)...",
        source_table, run_date, lookback_days,
    )

    query = f"""
        SELECT * EXCEPT (rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY hospital_id
                       ORDER BY report_date DESC
                   ) AS rn
            FROM `{source_table}`
            WHERE report_date >= DATE_SUB('{run_date}', INTERVAL {lookback_days} DAY)
              AND report_date <= '{run_date}'
        )
        WHERE rn = 1
        ORDER BY hospital_id
    """
    df = bq_client.query(query).to_dataframe()
    df["report_date"] = pd.to_datetime(df["report_date"])

    logger.info(
        "[QA] Loaded: rows=%d | hospitals=%d | report_date(latest)=%s | cols=%d",
        len(df),
        df["hospital_id"].nunique(),
        df["report_date"].max().date() if len(df) > 0 else "N/A",
        df.shape[1],
    )

    if len(df) == 0:
        raise ValueError(
            f"[LOAD FAIL] No feature data found for run_date={run_date} "
            f"with lookback={lookback_days} days. Check feature store pipeline."
        )

    # Sentinel cleanup: replace -9999 with NaN for imputation downstream
    for col in _SENTINEL_COLS:
        if col in df.columns:
            n = (df[col] == _SENTINEL_VALUE).sum()
            if n > 0:
                logger.warning(
                    "[QA] Column '%s': %d sentinel (-9999) values → NaN", col, n
                )
                df[col] = df[col].replace(_SENTINEL_VALUE, np.nan)

    return df

def load_model_artifacts(
    gcs_client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
    model_label: str,
) -> dict:
    """
    Load preprocessor, regression model, classifier, SHAP explainer,
    and feature_metadata from GCS prefix.

    Expected GCS layout:
        gs://<bucket>/<prefix>/preprocessor.pkl
        gs://<bucket>/<prefix>/model_reg.pkl
        gs://<bucket>/<prefix>/model_cls.pkl
        gs://<bucket>/<prefix>/explainer.pkl         (optional — rebuilt if absent)
        gs://<bucket>/<prefix>/feature_metadata.json (optional — fallback to preprocessor)

    model_version derived from model_reg.pkl blob generation number.
    """
    mandatory = {
        "preprocessor": f"{gcs_prefix}/preprocessor.pkl",
        "reg":          f"{gcs_prefix}/model_reg.pkl",
        "cls":          f"{gcs_prefix}/model_cls.pkl",
    }

    artifacts: dict = {}
    for key, blob_path in mandatory.items():
        if not _exists(gcs_client, bucket_name, blob_path):
            raise FileNotFoundError(
                f"[{model_label}] Required artifact missing: "
                f"gs://{bucket_name}/{blob_path}"
            )
        artifacts[key] = _load_pkl(gcs_client, bucket_name, blob_path)
        logger.info("[%s] Loaded %s ✓", model_label, blob_path)

    # Stable model version from blob generation
    artifacts["model_version"] = _get_blob_version(
        gcs_client, bucket_name, mandatory["reg"]
    )
    logger.info("[%s] model_version: %s", model_label, artifacts["model_version"])

    # SHAP explainer: load from GCS or rebuild from model
    artifacts["explainer"] = get_or_build_explainer(
        gcs_client,
        bucket_name,
        f"{gcs_prefix}/explainer.pkl",
        artifacts["reg"],
    )

    # Feature metadata: primary path for schema resolution
    meta_path = f"{gcs_prefix}/feature_metadata.json"
    if _exists(gcs_client, bucket_name, meta_path):
        artifacts["feature_metadata"] = _load_json(gcs_client, bucket_name, meta_path)
        logger.info(
            "[%s] feature_metadata.json loaded: %d features ✓",
            model_label,
            artifacts["feature_metadata"]["n_features"],
        )
    else:
        artifacts["feature_metadata"] = None
        logger.warning(
            "[%s] feature_metadata.json not found at gs://%s/%s — "
            "will fallback to preprocessor schema resolution.",
            model_label, bucket_name, meta_path,
        )

    return artifacts

def resolve_feature_schema(
    preprocessor,
    df: pd.DataFrame,
    feature_metadata: Optional[dict],
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve feature column ordering.

    Primary:  feature_metadata.json (deterministic, matches training)
    Fallback: preprocessor named_transformers_
    Last:     reconstruct from DataFrame columns (least reliable)

    Returns: (feature_cols, num_feats, cat_feats)
    """
    if feature_metadata is not None:
        num_feats    = feature_metadata["numeric_feats"]
        cat_feats    = feature_metadata["cat_feats"]
        feature_cols = num_feats + cat_feats
        logger.info(
            "Feature schema from feature_metadata.json: %d features (num=%d, cat=%d)",
            len(feature_cols), len(num_feats), len(cat_feats),
        )
    else:
        try:
            num_feats    = preprocessor.named_transformers_["num"].feature_names_in_.tolist()
            cat_feats    = preprocessor.named_transformers_["cat"].feature_names_in_.tolist()
            feature_cols = num_feats + cat_feats
            logger.warning(
                "Feature schema from preprocessor (fallback): %d features. "
                "Upload feature_metadata.json to GCS to use primary path.",
                len(feature_cols),
            )
        except Exception as e:
            logger.warning(
                "Cannot read schema from preprocessor (%s). Reconstructing from columns "
                "(least reliable — result may differ from training schema).", e,
            )
            feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
            cat_feats    = [c for c in _CATEGORICAL_COLS if c in feature_cols]
            num_feats    = [c for c in feature_cols if c not in _CATEGORICAL_COLS]

    # Backfill missing columns — imputer in preprocessor handles NaN
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(
            "[QA] %d feature columns absent from data → filled NaN: %s",
            len(missing), missing,
        )
        for col in missing:
            df[col] = np.nan
    else:
        logger.info("[QA] Column alignment: OK — all %d features present ✓", len(feature_cols))

    return feature_cols, num_feats, cat_feats

def run_inference(
    artifacts: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    all_feat_names: list[str],
    model_name: str,
    run_date: str,
    source_table: str,
) -> pd.DataFrame:
    preprocessor  = artifacts["preprocessor"]
    reg           = artifacts["reg"]
    cls           = artifacts["cls"]
    explainer     = artifacts["explainer"]
    model_version = artifacts.get("model_version", "unknown")

    # Transform
    logger.info("[%s] Transforming %d rows...", model_name, len(df))
    X = preprocessor.transform(df[feature_cols])

    # Assert shape before SHAP — detect preprocessor/schema mismatch early
    assert X.shape[1] == len(all_feat_names), (
        f"[SCHEMA MISMATCH] X.shape[1]={X.shape[1]} != "
        f"len(all_feat_names)={len(all_feat_names)}. "
        "Re-run data_prep.py and re-upload artifacts."
    )

    X_df = pd.DataFrame(X, columns=all_feat_names)
    logger.info("[%s] X shape: %s", model_name, X.shape)

    # Predict
    pred_occ  = reg.predict(X).clip(0, 1)
    pred_cls  = cls.predict(X)
    pred_prob = cls.predict_proba(X)[:, 1]

    logger.info(
        "[%s] Predictions: avg_occupancy=%.1f%% | high_strain_rate=%.1f%%",
        model_name, pred_occ.mean() * 100, pred_cls.mean() * 100,
    )

    # SHAP — topX_feature/topX_shap/topX_direction naming for _RAW_SCHEMA
    shap_vals, base_val = compute_shap_values(explainer, X_df)
    top_features = extract_top_n_features(shap_vals, all_feat_names, n=3)

    # Timezone-aware UTC timestamp (datetime.utcnow() deprecated in Python 3.12+)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records = []
    for i in range(len(df)):
        actual_occ = (
            float(df["target_occupancy_next_week"].iloc[i])
            if "target_occupancy_next_week" in df.columns else None
        )
        actual_strain = (
            int(df["target_high_strain"].iloc[i])
            if "target_high_strain" in df.columns else None
        )
        abs_err = (
            round(abs(actual_occ - pred_occ[i]), 4)
            if actual_occ is not None and not np.isnan(actual_occ) else None
        )

        record: dict = {
            "model_name":               model_name,
            "hospital_id":              str(df["hospital_id"].iloc[i]),
            "report_date":              str(df["report_date"].iloc[i])[:10],
            "run_date":                 run_date,
            "pred_occupancy_next_week": round(float(pred_occ[i]), 4),
            "pred_high_strain":         int(pred_cls[i]),
            "pred_high_strain_prob":    round(float(pred_prob[i]), 4),
            "actual_occupancy":         (
                round(actual_occ, 4)
                if actual_occ is not None and not np.isnan(actual_occ) else None
            ),
            "actual_high_strain":       actual_strain,
            "abs_error":                abs_err,
            "shap_base_value":          round(base_val, 4),
            "model_version":            model_version,
            "data_source":              source_table,
            "predicted_at":             now_utc,
        }
        # Injects: top1_feature, top1_shap, top1_direction, ... top3_*, explanation
        record.update(top_features[i])
        records.append(record)

    return pd.DataFrame(records)


def upload_raw_predictions(
    bq_client: bigquery.Client,
    df: pd.DataFrame,
    raw_table: str,
    run_date: str,
    write_disposition: str = "WRITE_APPEND",
) -> int:
    """
    Idempotent upload with explicit schema to vertex_batch_prediction_raw.

    """
    if write_disposition not in {"WRITE_APPEND", "WRITE_TRUNCATE"}:
        raise ValueError(f"Invalid write_disposition: {write_disposition}")

    if write_disposition == "WRITE_APPEND":
        delete_sql = f"DELETE FROM `{raw_table}` WHERE run_date = '{run_date}'"
        logger.info("Deleting run_date=%s from %s...", run_date, raw_table)
        try:
            bq_client.query(delete_sql).result()
        except Exception as e:
            logger.warning("Delete skipped (table may not exist yet): %s", e)

    # Coerce dtypes for BQ schema compatibility
    df["pred_high_strain"] = (
        pd.to_numeric(df["pred_high_strain"], errors="raise").astype("Int64")
    )
    df["actual_high_strain"] = (
        pd.to_numeric(df["actual_high_strain"], errors="coerce").astype("Int64")
    )
    df["report_date"]   = pd.to_datetime(df["report_date"]).dt.date
    df["run_date"]      = pd.to_datetime(df["run_date"]).dt.date
    df["predicted_at"]  = pd.to_datetime(df["predicted_at"], utc=True)

    logger.info(
        "Uploading %d records to %s (mode=%s)...",
        len(df), raw_table, write_disposition,
    )

    job_cfg = bigquery.LoadJobConfig(
        write_disposition=getattr(bigquery.WriteDisposition, write_disposition),
        schema=_RAW_SCHEMA,
    )
    job = bq_client.load_table_from_dataframe(df, raw_table, job_config=job_cfg)
    job.result()

    tbl = bq_client.get_table(raw_table)
    logger.info("Upload complete. %s now has %d total rows.", raw_table, tbl.num_rows)
    return tbl.num_rows

def log_inference_qa(df: pd.DataFrame) -> None:
    logger.info("=== Inference QA Summary ===")
    for model in df["model_name"].unique():
        s = df[df["model_name"] == model]
        hs_count = int(s["pred_high_strain"].sum())
        occ_gt_90 = (s["pred_occupancy_next_week"] >= 0.9).sum()
        logger.info(
            "[QA][%s] records=%d | avg_occupancy=%.1f%% | "
            "high_strain=%d (%.1f%%) | pred_occ>=90%%: %d (%.1f%%)",
            model, len(s),
            s["pred_occupancy_next_week"].mean() * 100,
            hs_count, hs_count / len(s) * 100,
            occ_gt_90, occ_gt_90 / len(s) * 100,
        )
        if "abs_error" in s.columns and s["abs_error"].notna().any():
            logger.info("[QA][%s] MAE=%.4f", model, s["abs_error"].mean())
    logger.info("=== End QA Summary ===")