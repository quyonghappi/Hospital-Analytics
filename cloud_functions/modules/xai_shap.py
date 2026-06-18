"""
xai_shap.py — SHAP utilities for hospital utilization inference pipeline.

Responsibilities:
  - Load/rebuild SHAP TreeExplainer from GCS or model object
  - Compute SHAP values for a transformed feature matrix
  - Extract top-N features per prediction row

Imported by: modules/inference.py
"""
import io
import logging
import pickle
from typing import Union

import numpy as np
import pandas as pd
import shap
from google.cloud import storage

logger = logging.getLogger(__name__)


def _load_pkl_from_gcs(gcs_client: storage.Client, bucket_name: str, blob_path: str):
    """Stream-download a pickle artifact from GCS and deserialize."""
    bucket = gcs_client.bucket(bucket_name)
    buf = io.BytesIO()
    bucket.blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pickle.load(buf)


def _blob_exists(gcs_client: storage.Client, bucket_name: str, blob_path: str) -> bool:
    return gcs_client.bucket(bucket_name).blob(blob_path).exists()


def get_or_build_explainer(
    gcs_client: storage.Client,
    bucket_name: str,
    explainer_blob_path: str,
    model,
) -> shap.TreeExplainer:
    """
    Load pre-fitted TreeExplainer from GCS if present; rebuild from model otherwise.

    Rebuilding is ~3x slower but a safe fallback when artifacts are re-trained
    and explainer.pkl hasn't been regenerated yet.
    """
    if _blob_exists(gcs_client, bucket_name, explainer_blob_path):
        try:
            explainer = _load_pkl_from_gcs(gcs_client, bucket_name, explainer_blob_path)
            logger.info(
                "SHAP explainer loaded from GCS: gs://%s/%s",
                bucket_name, explainer_blob_path,
            )
            return explainer
        except Exception as e:
            logger.warning(
                "Failed to deserialize explainer from GCS (%s). Will rebuild. Error: %s",
                explainer_blob_path, e,
            )
    else:
        logger.warning(
            "explainer.pkl not found at gs://%s/%s — rebuilding TreeExplainer from model.",
            bucket_name, explainer_blob_path,
        )

    logger.info("Building TreeExplainer from fitted model...")
    return shap.TreeExplainer(model)


# ─── SHAP Computation ─────────────────────────────────────────────────────────

def compute_shap_values(
    explainer: shap.TreeExplainer,
    X_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for the transformed feature matrix.

    Handles both regression (returns ndarray) and binary classification
    (returns list of arrays — we take index [1] for the positive class).

    Returns:
        shap_matrix : np.ndarray, shape (n_samples, n_features)
        base_value  : float, explainer expected value
    """
    logger.info("Computing SHAP values for %d samples × %d features...", *X_df.shape)
    sv = explainer.shap_values(X_df)

    # Binary classifier explainers return a list [neg_class, pos_class]
    if isinstance(sv, list):
        sv = sv[1]

    raw_base = explainer.expected_value
    base_val = float(raw_base[0]) if np.ndim(raw_base) > 0 else float(raw_base)

    logger.info("SHAP complete. Matrix: %s | base_value: %.4f", sv.shape, base_val)
    return sv, base_val


# ─── Top-N Feature Extraction ─────────────────────────────────────────────────

def extract_top_n_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 3,
) -> list[dict]:
    """
    For each prediction row, extract the top-N features by absolute SHAP contribution.

    Output dict keys (for n=3):
        top1_feature, top1_shap, top1_direction,
        top2_feature, top2_shap, top2_direction,
        top3_feature, top3_shap, top3_direction,
        explanation   (human-readable summary string)

    Column naming matches _RAW_SCHEMA in inference.py exactly.
    post_process.py renames topX_* → shap_feature_X when projecting to ml_forecast_results.
    """
    results = []
    for i in range(len(shap_values)):
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:n]

        row: dict = {}
        summary_parts = []
        for rank, idx in enumerate(top_idx, start=1):
            direction = "up" if sv[idx] > 0 else "down"
            row[f"top{rank}_feature"]   = feature_names[idx]
            row[f"top{rank}_shap"]      = round(float(sv[idx]), 4)
            row[f"top{rank}_direction"] = direction
            summary_parts.append(
                f"{feature_names[idx]} ({direction} {abs(sv[idx]):.3f})"
            )

        row["explanation"] = ", ".join(summary_parts)
        results.append(row)

    return results