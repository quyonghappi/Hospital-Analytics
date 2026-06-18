"""
data_prep.py — ML Data Preparation for Hospital Utilization Pipeline.
Usage:
  python src/data_prep.py \
      --project_id <PROJECT_ID> \
      --fs_dataset hospital_feature_store \
      --fs_table fs_hospital_weekly \
      --train_end 2023-12-31 \
      --val_end 2024-03-31 \
      --artifact_dir /app/ml_artifacts
"""
# NEXT RETRAIN DECISION (when new HHS data arrives):
#   Option A: Keep OrdinalEncoder — simpler, no leakage concern
#   Option B: Switch to TargetEncoder(cv=5) — data_prep.py original design
#   → Benchmark cả hai trên val set trước khi quyết định
import json
import logging
import os
import pickle
import argparse
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder  # [FIX-3]
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("data_prep")

_LEAKAGE_KEYWORDS = ["_next_week", "target_", "_next_", "next_week_"]

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital ML Data Preparation")
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--fs_dataset", type=str, default="hospital_feature_store")
    parser.add_argument("--fs_table", type=str, default="fs_hospital_weekly")
    parser.add_argument("--train_end", type=str, default="2022-12-31")
    parser.add_argument("--val_end", type=str, default="2023-09-30")
    parser.add_argument("--artifact_dir", type=str, default="/app/ml_artifacts")
    # nếu split quá nhỏ thì pipeline fail
    # thay vì âm thầm train trên data không đủ
    parser.add_argument("--min_rows_per_split", type=int, default=500)
    return parser.parse_args()


# Data Loading
def load_feature_store(
    bq_client: bigquery.Client, project_id: str, dataset: str, table: str
) -> pd.DataFrame:
    logger.info("Loading feature store from %s.%s.%s...", project_id, dataset, table)
    query = (
        f"SELECT * FROM `{project_id}.{dataset}.{table}` "
        f"ORDER BY hospital_id, report_date"
    )
    df = bq_client.query(query).to_dataframe()
    df["report_date"] = pd.to_datetime(df["report_date"])
    logger.info(
        "[QA] Loaded: rows=%d | hospitals=%d | date_range=%s → %s",
        len(df),
        df["hospital_id"].nunique(),
        df["report_date"].min().date(),
        df["report_date"].max().date(),
    )
    return df


# Data Validation
def validate_feature_store(df: pd.DataFrame, min_rows: int) -> None:
    """
    Checks:
      - Row count tối thiểu (guard against empty / near-empty feature store)
      - Null rate trên identifier columns (hospital_id, report_date)
      - Null rate trên target columns
      - Duplicate (hospital_id, report_date) pairs
    """
    logger.info("=== Pre-split Data Validation ===")

    assert len(df) >= min_rows, (
        f"[VALIDATION FAIL] Feature store has only {len(df)} rows — "
        f"minimum required: {min_rows}. Abort."
    )
    logger.info("[QA] Row count: %d ✓", len(df))

    # Null check on identifiers
    id_cols = ["hospital_id", "report_date"]
    for col in id_cols:
        null_rate = df[col].isna().mean()
        assert null_rate == 0.0, (
            f"[VALIDATION FAIL] Identifier column '{col}' has {null_rate:.1%} null values. "
            f"Cannot proceed — data integrity broken."
        )
    logger.info("[QA] Identifier null check: OK ✓")

    # Null check on targets - missing rate < 5%
    target_cols = ["target_occupancy_next_week", "target_high_strain"]
    for col in [c for c in target_cols if c in df.columns]:
        null_rate = df[col].isna().mean()
        if null_rate > 0.05:
            logger.warning(
                "[QA] Target column '%s' null rate=%.1%% — exceeds NFR-DQ-01 threshold (5%%)",
                col, null_rate,
            )
        else:
            logger.info("[QA] Target '%s' null rate=%.2f%% ✓", col, null_rate * 100)

    # duplicate (hospital_id, report_date)
    n_dupes = df.duplicated(subset=["hospital_id", "report_date"]).sum()
    if n_dupes > 0:
        logger.warning(
            "[QA] %d duplicate (hospital_id, report_date) rows detected — "
            "will keep first occurrence. Investigate upstream feature store.",
            n_dupes,
        )
        df.drop_duplicates(subset=["hospital_id", "report_date"], keep="first", inplace=True)
    else:
        logger.info("[QA] Duplicate check: OK ✓")

    logger.info("=== Validation Complete ===")

def perform_temporal_split(
    df: pd.DataFrame, train_end: str, val_end: str, min_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["report_date"] <= train_end].copy()
    val_df   = df[(df["report_date"] > train_end) & (df["report_date"] <= val_end)].copy()
    test_df  = df[df["report_date"] > val_end].copy()

    logger.info(
        "Temporal split: train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    for name, split in [("val", val_df), ("test", test_df)]:
        assert len(split) >= min_rows, (
            f"[SPLIT FAIL] '{name}' split has only {len(split)} rows — "
            f"minimum required: {min_rows}. Check train_end / val_end arguments."
        )

    return train_df, val_df, test_df


def assert_no_leakage(feature_cols: list[str]) -> None:
    """
    Reject feature columns that contain target-related keywords.
    """
    violations = [
        col for col in feature_cols
        if any(kw in col.lower() for kw in _LEAKAGE_KEYWORDS)
    ]
    assert not violations, (
        f"[LEAKAGE GUARD] Potential leakage columns detected in feature_cols: {violations}\n"
        f"  → Remove these columns from the feature store query, or rename if they are "
        f"legitimate lag features (e.g. 'lag_1_occ_rate' instead of 'occupancy_rate_lag1')."
    )
    logger.info("[LEAKAGE GUARD] No leakage-risk columns detected ✓")


def build_preprocessor(numeric_feats: list[str], cat_feats: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    return ColumnTransformer(
        [("num", num_pipe, numeric_feats), ("cat", cat_pipe, cat_feats)],
        remainder="drop",
    )


# Artifact Export
def save_parquet_artifacts(
    artifact_dir: str, dfs: dict[str, pd.DataFrame], save_cols: list[str]
) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    for name, df in dfs.items():
        valid_cols = [c for c in save_cols if c in df.columns]
        path = os.path.join(artifact_dir, f"{name}_features.parquet")
        df[valid_cols].to_parquet(path, index=False)
        logger.info("Saved parquet artifact: %s (%d rows)", path, len(df))


def save_feature_metadata(
    artifact_dir: str,
    numeric_feats: list[str],
    cat_feats: list[str],
    train_end: str,
    val_end: str,
    split_counts: dict[str, int],
) -> None:
    """
     feature schema + split metadata cùng preprocessor.pkl.
      - inference.py load file này để resolve feature_cols deterministically,
        thay vì fallback reconstruct từ DataFrame columns (fragile).
      - Audit trail: biết preprocessor này được fit trên data range nào,
        với bao nhiêu rows.
      - Model registry: khi compare model versions, biết chính xác feature list
        của từng version.
    """
    metadata = {
        "numeric_feats": numeric_feats,
        "cat_feats": cat_feats,
        "feature_cols": numeric_feats + cat_feats,
        "train_end": train_end,
        "val_end": val_end,
        "split_counts": split_counts,
        "n_features": len(numeric_feats) + len(cat_feats),
    }
    path = os.path.join(artifact_dir, "feature_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved feature metadata: %s", path)

def arrays_to_bq(
    bq_client: bigquery.Client,
    project_id: str,
    dataset: str,
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    split_name: str,
    feat_names: list[str],
    hospital_ids: pd.Series,
    report_dates: pd.Series,
) -> None:
    feat_df = pd.DataFrame(X, columns=feat_names)

    # Identity columns — cần thiết để join lại với actuals khi evaluate
    feat_df["hospital_id"]  = hospital_ids.values
    feat_df["report_date"]  = report_dates.dt.strftime("%Y-%m-%d").values
    feat_df["target_occupancy_next_week"] = y_reg
    feat_df["target_high_strain"] = y_cls
    feat_df["split"] = split_name

    table_id = f"{project_id}.{dataset}.ml_{split_name}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(feat_df, table_id, job_config=job_config).result()
    logger.info("Uploaded BQ table: %s → %d rows", table_id, len(feat_df))

def main() -> None:
    args = get_args()
    bq = bigquery.Client(project=args.project_id)

    fs_df = load_feature_store(bq, args.project_id, args.fs_dataset, args.fs_table)

    validate_feature_store(fs_df, min_rows=args.min_rows_per_split)

    train_df, val_df, test_df = perform_temporal_split(
        fs_df, args.train_end, args.val_end, min_rows=args.min_rows_per_split
    )

    _EXCLUDE = {
        "hospital_id", "report_date", "county_fips", "zip_code", "county_name",
        "hospital_name", "city", "feature_computed_at",
        "target_occupancy_next_week", "target_high_strain", "occupancy_rate",
    }
    _CATEGORICALS = {
        "state", "hospital_type", "season", "disease_season",
        "healthcare_risk_level", "metro_nonmetro_flag", "hrr_region",
    }

    feature_cols = [c for c in fs_df.columns if c not in _EXCLUDE]
    cat_feats    = [c for c in _CATEGORICALS if c in feature_cols]
    numeric_feats = [c for c in feature_cols if c not in _CATEGORICALS]
    all_feat_names = numeric_feats + cat_feats

    assert_no_leakage(feature_cols)

    preprocessor = build_preprocessor(numeric_feats, cat_feats)
    logger.info("Fitting preprocessor on train set (%d rows)...", len(train_df))
    X_train = preprocessor.fit_transform(
        train_df[feature_cols],
        # train_df["target_occupancy_next_week"].clip(0, 1),  # y for TargetEncoder
    )
    X_val  = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    y_tr_reg = train_df["target_occupancy_next_week"].clip(0, 1).values
    y_va_reg = val_df["target_occupancy_next_week"].clip(0, 1).values
    y_te_reg = test_df["target_occupancy_next_week"].clip(0, 1).values

    y_tr_cls = train_df["target_high_strain"].values.astype(int)
    y_va_cls = val_df["target_high_strain"].values.astype(int)
    y_te_cls = test_df["target_high_strain"].values.astype(int)

    os.makedirs(args.artifact_dir, exist_ok=True)

    # preprocessor.pkl
    with open(os.path.join(args.artifact_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Saved preprocessor.pkl")

    # feature_metadata.json
    save_feature_metadata(
        artifact_dir=args.artifact_dir,
        numeric_feats=numeric_feats,
        cat_feats=cat_feats,
        train_end=args.train_end,
        val_end=args.val_end,
        split_counts={
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        },
    )

    # Parquet snapshots (raw, trước transform) để debug và re-run
    save_cols = (
        feature_cols
        + ["target_occupancy_next_week", "target_high_strain", "hospital_id", "report_date"]
    )
    save_parquet_artifacts(
        args.artifact_dir,
        {"train": train_df, "val": val_df, "test": test_df},
        save_cols,
    )

    for split_name, X, y_reg, y_cls, split_df in [
        ("train", X_train, y_tr_reg, y_tr_cls, train_df),
        ("val",   X_val,   y_va_reg, y_va_cls, val_df),
        ("test",  X_test,  y_te_reg, y_te_cls, test_df),
    ]:
        arrays_to_bq(
            bq_client=bq,
            project_id=args.project_id,
            dataset=args.fs_dataset,
            X=X,
            y_reg=y_reg,
            y_cls=y_cls,
            split_name=split_name,
            feat_names=all_feat_names,
            hospital_ids=split_df["hospital_id"].reset_index(drop=True),
            report_dates=split_df["report_date"].reset_index(drop=True),
        )

    logger.info("data_prep.py complete.")


if __name__ == "__main__":
    main()