"""
post_process.py — Post-processing: vertex_batch_prediction_raw → serving tables.

SoC responsibility:
  - Input:  ml_predictions_dev.vertex_batch_prediction_raw   (raw model output)
  - Output: ml_observability.model_predictions_shap          (MLOps / monitoring)
            ml_predictions_dev.ml_forecast_results           (Looker Studio + alert engine)

Business logic lives HERE, not in inference.py:
  - alert_flag          = pred_occupancy_next_week >= 0.90
  - forecast_date       = report_date + INTERVAL 7 DAY
  - confidence_lower/upper = prediction ± 0.05 (placeholder)
  - hospital_name, state = JOIN dim_hospitals
  - shap_feature_1/2/3  = renamed from top1_feature/top2_feature/top3_feature
  - shap_value_1/2/3    = renamed from top1_shap/top2_shap/top3_shap
  - Only XGBoost rows → ml_forecast_results (primary serving model)

Both projections are idempotent: DELETE WHERE run_date = X → INSERT.

Usage:
  python src/post_process.py \\
      --project_id <PROJECT_ID> \\
      --raw_table <PROJECT.DATASET.vertex_batch_prediction_raw> \\
      --observability_table <PROJECT.DATASET.model_predictions_shap> \\
      --forecast_table <PROJECT.DATASET.ml_forecast_results> \\
      --dim_table <PROJECT.DATASET.dim_hospitals> \\
      --run_date 2024-06-01

Airflow integration:
  post_process chạy sau inference (task dependency: inference >> post_process).
  Nếu inference fail, post_process không chạy → serving tables giữ nguyên data cũ.
  Idempotent: re-run post_process với cùng run_date an toàn.
"""

import argparse
import logging
import sys
from datetime import date

from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("post_process")


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_raw_table(
    bq_client: bigquery.Client,
    raw_table: str,
    run_date: str,
    min_rows: int = 1,
) -> int:
    """
    Guard: đảm bảo vertex_batch_prediction_raw có data cho run_date trước khi
    downstream tables bị xoá và re-populated.

    Fail fast: nếu raw table trống → downstream tables không bị xoá.
    """
    result = bq_client.query(f"""
        SELECT COUNT(*) AS n, COUNT(DISTINCT hospital_id) AS n_hospitals
        FROM `{raw_table}`
        WHERE run_date = '{run_date}'
    """).to_dataframe()

    n_rows      = int(result["n"].iloc[0])
    n_hospitals = int(result["n_hospitals"].iloc[0])

    if n_rows < min_rows:
        raise ValueError(
            f"[VALIDATION FAIL] {raw_table} has {n_rows} rows for run_date={run_date}. "
            f"Minimum required: {min_rows}. "
            f"inference.py may have failed or not been run yet."
        )

    logger.info(
        "[QA] Raw table validated: run_date=%s | rows=%d | hospitals=%d ✓",
        run_date, n_rows, n_hospitals,
    )
    return n_rows


# ─── Projection 1: model_predictions_shap (MLOps / Observability) ─────────────

def project_to_observability(
    bq_client: bigquery.Client,
    raw_table: str,
    observability_table: str,
    run_date: str,
) -> int:
    """
    vertex_batch_prediction_raw → ml_observability.model_predictions_shap

    Transformations:
      - SAFE_CAST type coercion trên tất cả columns
      - Không rename SHAP columns (top1_feature naming được giữ nguyên cho MLOps)
      - Bao gồm ALL model_name values (XGBoost + LightGBM) để model comparison

    Không có business logic ở đây — đây là observability table phục vụ:
      - Model drift monitoring
      - SHAP feature importance tracking
      - Prediction distribution analysis
      - Actual vs predicted backfill (luồng riêng)
    """
    logger.info(
        "Projecting run_date=%s → observability: %s", run_date, observability_table
    )

    # Idempotent: xóa run_date partition trước
    _delete_run_date(bq_client, observability_table, run_date, "observability")

    sql = f"""
        INSERT INTO `{observability_table}` (
            model_name,
            hospital_id,
            report_date,
            run_date,
            pred_occupancy_next_week,
            pred_high_strain,
            pred_high_strain_prob,
            actual_occupancy,
            actual_high_strain,
            abs_error,
            shap_base_value,
            top1_feature,
            top1_shap,
            top1_direction,
            top2_feature,
            top2_shap,
            top2_direction,
            top3_feature,
            top3_shap,
            top3_direction,
            explanation
        )
        SELECT
            CAST(model_name               AS STRING)  AS model_name,
            CAST(hospital_id              AS STRING)  AS hospital_id,
            SAFE_CAST(report_date         AS DATE)    AS report_date,
            CAST(run_date                 AS DATE)    AS run_date,
            CAST(pred_occupancy_next_week AS FLOAT64) AS pred_occupancy_next_week,
            CAST(pred_high_strain         AS INT64)   AS pred_high_strain,
            CAST(pred_high_strain_prob    AS FLOAT64) AS pred_high_strain_prob,
            -- actual_* NULL tại inference time — backfill bằng luồng riêng
            CAST(actual_occupancy         AS FLOAT64) AS actual_occupancy,
            CAST(actual_high_strain       AS INT64)   AS actual_high_strain,
            CAST(abs_error                AS FLOAT64) AS abs_error,
            CAST(shap_base_value          AS FLOAT64) AS shap_base_value,
            CAST(top1_feature             AS STRING)  AS top1_feature,
            CAST(top1_shap                AS FLOAT64) AS top1_shap,
            CAST(top1_direction           AS STRING)  AS top1_direction,
            CAST(top2_feature             AS STRING)  AS top2_feature,
            CAST(top2_shap                AS FLOAT64) AS top2_shap,
            CAST(top2_direction           AS STRING)  AS top2_direction,
            CAST(top3_feature             AS STRING)  AS top3_feature,
            CAST(top3_shap                AS FLOAT64) AS top3_shap,
            CAST(top3_direction           AS STRING)  AS top3_direction,
            CAST(explanation              AS STRING)  AS explanation
        FROM `{raw_table}`
        WHERE run_date = '{run_date}'
    """

    bq_client.query(sql).result()

    n = _count_rows(bq_client, observability_table, run_date)
    logger.info(
        "Observability projection complete: %d rows inserted for run_date=%s ✓",
        n, run_date,
    )
    return n


# ─── Projection 2: ml_forecast_results (Serving / Looker Studio + Alert) ──────

def project_to_forecast_results(
    bq_client: bigquery.Client,
    raw_table: str,
    forecast_table: str,
    dim_table: str,
    run_date: str,
) -> int:
    """
    vertex_batch_prediction_raw → ml_predictions_dev.ml_forecast_results

    Business logic applied here (NOT in inference.py):

      forecast_date        = report_date + 7 days               (next-week prediction target)
      alert_flag           = pred_occupancy_next_week >= 0.90    (spec §5.6 SFR-07)
      confidence_lower_95  = MAX(0.0, prediction - 0.05)        (placeholder — replace với
      confidence_upper_95  = MIN(1.0, prediction + 0.05)         quantile regression / conformal)
      hospital_name, state = LEFT JOIN dim_hospitals             (dim attributes, not in raw)
      shap_feature_1/2/3   = top1_feature/top2_feature/top3_feature (rename for serving schema)
      shap_value_1/2/3     = top1_shap/top2_shap/top3_shap          (rename for serving schema)

    Chỉ lấy XGBoost rows — primary model cho serving table.
    LightGBM rows chỉ tồn tại trong model_predictions_shap để comparison.

    TODO: thêm predicted_patient_volume, predicted_los khi model có regression heads.
    """
    logger.info(
        "Projecting run_date=%s → forecast results: %s (dim: %s)",
        run_date, forecast_table, dim_table,
    )

    # Idempotent: xóa run_date partition trước
    _delete_run_date(bq_client, forecast_table, run_date, "forecast_results")

    sql = f"""
        INSERT INTO `{forecast_table}` (
            forecast_id,
            hospital_id,
            hospital_name,
            state,
            run_date,
            forecast_date,
            predicted_occupancy_rate,
            confidence_lower_95,
            confidence_upper_95,
            alert_flag,
            shap_feature_1,
            shap_feature_2,
            shap_feature_3,
            shap_value_1,
            shap_value_2,
            shap_value_3,
            model_version
        )
        SELECT
            GENERATE_UUID()                                             AS forecast_id,
            p.hospital_id,
            d.hospital_name,
            d.state,
            p.run_date,
            -- forecast_date = 7 ngày sau report_date (next-week prediction target)
            DATE_ADD(SAFE_CAST(p.report_date AS DATE), INTERVAL 7 DAY) AS forecast_date,
            p.pred_occupancy_next_week                                  AS predicted_occupancy_rate,
            -- Confidence intervals: ±0.05 approximation quanh prediction.
            -- TODO: thay bằng model-native intervals (quantile regression / conformal prediction).
            GREATEST(0.0, p.pred_occupancy_next_week - 0.05)           AS confidence_lower_95,
            LEAST(1.0,    p.pred_occupancy_next_week + 0.05)           AS confidence_upper_95,
            -- alert_flag: TRUE nếu predicted >= 90% occupancy (spec §5.6 SFR-07)
            (p.pred_occupancy_next_week >= 0.90)                        AS alert_flag,
            -- SHAP columns: rename topX_feature → shap_feature_X cho serving schema
            p.top1_feature  AS shap_feature_1,
            p.top2_feature  AS shap_feature_2,
            p.top3_feature  AS shap_feature_3,
            p.top1_shap     AS shap_value_1,
            p.top2_shap     AS shap_value_2,
            p.top3_shap     AS shap_value_3,
            p.model_version
        FROM `{raw_table}` p
        LEFT JOIN `{dim_table}` d ON p.hospital_id = d.hospital_id
        WHERE p.run_date   = '{run_date}'
          AND p.model_name = 'XGBoost'
    """

    bq_client.query(sql).result()

    n = _count_rows(bq_client, forecast_table, run_date)
    logger.info(
        "Forecast projection complete: %d rows inserted for run_date=%s ✓",
        n, run_date,
    )

    # Log alert count — operational visibility
    alert_result = bq_client.query(f"""
        SELECT COUNT(*) AS n_alerts
        FROM `{forecast_table}`
        WHERE run_date = '{run_date}' AND alert_flag = TRUE
    """).to_dataframe()
    n_alerts = int(alert_result["n_alerts"].iloc[0])
    logger.info(
        "[QA] alert_flag=TRUE: %d hospitals in run_date=%s (threshold: occ >= 90%%)",
        n_alerts, run_date,
    )

    return n


# ─── QA Summary ───────────────────────────────────────────────────────────────

def log_post_process_qa(
    bq_client: bigquery.Client,
    raw_table: str,
    observability_table: str,
    forecast_table: str,
    run_date: str,
) -> None:
    """
    Row count reconciliation: raw → observability, raw → forecast.
    Nếu counts không khớp → upstream data issue hoặc JOIN problem.
    """
    logger.info("=== Post-Process QA ===")

    raw_counts = bq_client.query(f"""
        SELECT model_name, COUNT(*) AS n
        FROM `{raw_table}`
        WHERE run_date = '{run_date}'
        GROUP BY model_name
    """).to_dataframe()
    for _, row in raw_counts.iterrows():
        logger.info("[QA] Raw %-10s: %d rows", row["model_name"], row["n"])

    obs_n      = _count_rows(bq_client, observability_table, run_date)
    forecast_n = _count_rows(bq_client, forecast_table,      run_date)
    raw_total  = int(raw_counts["n"].sum())
    raw_xgb    = int(raw_counts.loc[raw_counts["model_name"] == "XGBoost", "n"].sum()) if not raw_counts.empty else 0

    logger.info("[QA] Observability rows  : %d (expected: %d / all models)", obs_n, raw_total)
    logger.info("[QA] Forecast rows       : %d (expected: %d / XGBoost only)", forecast_n, raw_xgb)

    if obs_n != raw_total:
        logger.warning(
            "[QA] Observability row count mismatch: got %d, expected %d. "
            "Check for INSERT errors or schema mismatches.",
            obs_n, raw_total,
        )
    if forecast_n != raw_xgb:
        logger.warning(
            "[QA] Forecast row count mismatch: got %d, expected %d XGBoost rows. "
            "Check dim_hospitals JOIN — hospitals missing in dim will still appear (LEFT JOIN).",
            forecast_n, raw_xgb,
        )

    logger.info("=== End Post-Process QA ===")


# ─── BQ Helpers ───────────────────────────────────────────────────────────────

def _delete_run_date(
    bq_client: bigquery.Client,
    table: str,
    run_date: str,
    label: str,
) -> None:
    try:
        bq_client.query(
            f"DELETE FROM `{table}` WHERE run_date = '{run_date}'"
        ).result()
        logger.info("[%s] Deleted run_date=%s from %s", label, run_date, table)
    except Exception as e:
        logger.warning(
            "[%s] Delete skipped (table may not exist yet): %s", label, e
        )


def _count_rows(
    bq_client: bigquery.Client,
    table: str,
    run_date: str,
) -> int:
    result = bq_client.query(
        f"SELECT COUNT(*) AS n FROM `{table}` WHERE run_date = '{run_date}'"
    ).to_dataframe()
    return int(result["n"].iloc[0])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-process vertex_batch_prediction_raw into serving tables"
    )
    p.add_argument("--project_id",          required=True)
    p.add_argument("--raw_table",           required=True,
                   help="Source: project.dataset.vertex_batch_prediction_raw")
    p.add_argument("--observability_table", required=True,
                   help="Sink: project.dataset.model_predictions_shap (MLOps)")
    p.add_argument("--forecast_table",      required=True,
                   help="Sink: project.dataset.ml_forecast_results (Looker Studio / Alerts)")
    p.add_argument("--dim_table",           required=True,
                   help="Dim: project.dataset.dim_hospitals (for hospital_name, state JOIN)")
    p.add_argument("--run_date",            default=str(date.today()),
                   help="Partition to process (YYYY-MM-DD). Default: today.")
    p.add_argument("--min_raw_rows",        type=int, default=1,
                   help="Minimum rows expected in raw_table for run_date (guard against empty)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("Post-Process START")
    logger.info("  run_date            : %s", args.run_date)
    logger.info("  raw_table           : %s", args.raw_table)
    logger.info("  observability_table : %s", args.observability_table)
    logger.info("  forecast_table      : %s", args.forecast_table)
    logger.info("  dim_table           : %s", args.dim_table)
    logger.info("=" * 60)

    bq_client = bigquery.Client(project=args.project_id)

    # 1. Validate raw table có data — fail fast trước khi xóa serving tables
    validate_raw_table(
        bq_client, args.raw_table, args.run_date, min_rows=args.min_raw_rows
    )

    # 2. Project → model_predictions_shap (observability, all models)
    project_to_observability(
        bq_client=bq_client,
        raw_table=args.raw_table,
        observability_table=args.observability_table,
        run_date=args.run_date,
    )

    # 3. Project → ml_forecast_results (serving, XGBoost only + business logic)
    project_to_forecast_results(
        bq_client=bq_client,
        raw_table=args.raw_table,
        forecast_table=args.forecast_table,
        dim_table=args.dim_table,
        run_date=args.run_date,
    )

    # 4. QA reconciliation
    log_post_process_qa(
        bq_client=bq_client,
        raw_table=args.raw_table,
        observability_table=args.observability_table,
        forecast_table=args.forecast_table,
        run_date=args.run_date,
    )

    logger.info(
        "Post-Process COMPLETE | run_date=%s | observability=%s | forecast=%s",
        args.run_date, args.observability_table, args.forecast_table,
    )


if __name__ == "__main__":
    main()