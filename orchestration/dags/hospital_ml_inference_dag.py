from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryInsertJobOperator,
)
from airflow.utils.trigger_rule import TriggerRule

from config import (
    PROJECT_ID,
    REGION,
    CF_INFERENCE_URL,   # New: Cloud Function HTTP trigger URL
    INFER_FS_TABLE,
    RAW_TABLE,
    FORECAST_TABLE,
    OBSERVABILITY_TABLE,
    DIM_TABLE,
    GCP_CONN_ID,
)

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":             "ml-team",
    "depends_on_past":   False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=20),
    "execution_timeout": timedelta(hours=3),
    # TODO: replace with Slack/PagerDuty callback
    # "on_failure_callback": slack_alert,
}


# ─── CF Trigger ──────────────────────────────────────────────────────────────

def _trigger_inference_cf(**context: Any) -> dict:
    """
    HTTP POST to Cloud Function inference worker. Blocks until CF responds.

    Auth: OIDC identity token scoped to CF_INFERENCE_URL audience.
    The token is generated from the Composer environment's service account identity.

    On CF 500 → AirflowException → Airflow marks task FAILED → retries per retry config.
    On CF 400 → AirflowException (no retry — bad payload is a DAG bug, not transient).

    Note on timeout: CF Gen 2 max timeout is 60 min. requests.post timeout is set to
    3600s as ceiling. Airflow task execution_timeout (90min) provides the outer bound.
    The Airflow task should never reach its timeout in normal operation.
    """
    # Lazy imports: these are available in Cloud Composer but avoiding top-level import
    # keeps the DAG parse step fast (Composer parses all DAGs on startup).
    import requests
    import google.oauth2.id_token
    from google.auth.transport.requests import Request as GoogleAuthRequest

    run_date      = context["ds"]
    lookback_days = context["params"]["lookback_days"]

    payload = {
        "run_date":      run_date,
        "lookback_days": lookback_days,
    }

    log.info("[CF Trigger] POST %s | payload=%s", CF_INFERENCE_URL, payload)

    # OIDC token: audience MUST be the exact CF URL (not https://www.googleapis.com/auth/cloud-platform)
    auth_req = GoogleAuthRequest()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, CF_INFERENCE_URL)

    try:
        resp = requests.post(
            CF_INFERENCE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=3600,  # 60min CF max + buffer — Airflow task timeout is outer guard
        )
    except requests.Timeout:
        raise AirflowException(
            f"CF inference request timed out after 3600s for run_date={run_date}. "
            "Check CF Cloud Logging for partial execution state."
        )
    except requests.ConnectionError as e:
        raise AirflowException(f"CF inference connection error: {e}") from e

    # Parse response regardless of status code for error detail
    try:
        result = resp.json()
    except Exception:
        result = {"raw_body": resp.text}

    if resp.status_code == 400:
        # 4xx = bad payload from this DAG (code bug) — don't retry, fix the DAG
        raise AirflowException(
            f"CF inference returned 400 (bad request — DAG payload bug): "
            f"[{result.get('error_type')}] {result.get('message')}"
        )

    if resp.status_code >= 500 or result.get("status") == "error":
        # 5xx = CF pipeline failure — Airflow will retry per task retry config
        raise AirflowException(
            f"CF inference failed (HTTP {resp.status_code}): "
            f"[{result.get('error_type', 'Unknown')}] {result.get('message', resp.text)}"
        )

    log.info(
        "[CF Trigger] Inference complete | run_date=%s | records=%s | elapsed=%.1fs",
        result.get("run_date"),
        result.get("records"),
        result.get("elapsed_sec", 0),
    )
    return result


# ─── HHS Skip Check ──────────────────────────────────────────────────────────

def _has_new_feature_data(**context: Any) -> bool:
    """
    Short-circuit when feature store has no new data since last inference run.

    HHS publishes data every 3-4 weeks, not every Monday. ~75% of DAG runs
    are no-ops. This guard prevents redundant CF invocations and duplicate
    run_date partitions in the raw table.

    Return False → all downstream tasks = SKIPPED → DAG run = SUCCESS (expected behavior).
    Return True  → proceed with CF trigger.
    """
    run_date = context["ds"]
    hook = BigQueryHook(
        gcp_conn_id=GCP_CONN_ID,
        use_legacy_sql=False,
        location=REGION,
        project_id=PROJECT_ID,
    )

    # Latest snapshot available in feature store up to run_date
    fs_max = hook.get_first(
        f"SELECT MAX(report_date) FROM `{INFER_FS_TABLE}` "
        f"WHERE report_date <= '{run_date}'"
    )[0]

    if fs_max is None:
        log.warning("[SkipCheck] Feature store empty ≤ %s. Skipping.", run_date)
        return False

    # Latest report_date already processed in previous runs (exclude current run_date)
    raw_max_row = hook.get_first(
        f"SELECT MAX(report_date) FROM `{RAW_TABLE}` "
        f"WHERE run_date < '{run_date}'"
    )
    raw_max = raw_max_row[0] if raw_max_row else None

    if raw_max is not None and fs_max <= raw_max:
        log.info(
            "[SkipCheck] No new HHS data — fs_max=%s already processed (raw_max=%s). Skipping CF.",
            fs_max, raw_max,
        )
        return False

    log.info(
        "[SkipCheck] New data — fs_max=%s > last_processed=%s. Triggering CF.",
        fs_max, raw_max,
    )
    return True


# ─── Post-processing SQL ──────────────────────────────────────────────────────
# Business logic is defined HERE, not inside the CF:
#   alert_flag           = pred_occupancy_next_week >= 0.90   (spec §5.6 SFR-07)
#   forecast_date        = report_date + INTERVAL 7 DAY
#   confidence_lower/upper = prediction ± 0.05 (placeholder — replace with conformal)
#   hospital_name/state  = LEFT JOIN dim_hospital
#   shap_feature_X       = renamed from topX_feature (serving schema convention)
#
# Benefit: changing a threshold (e.g. 0.90 → 0.85) is a SQL edit + DAG deploy,
# NOT a CF rebuild + push + deploy.

_sql_observability = f"""
    DELETE FROM `{OBSERVABILITY_TABLE}` WHERE run_date = '{{{{ ds }}}}';

    INSERT INTO `{OBSERVABILITY_TABLE}` (
        model_name, hospital_id, report_date, run_date,
        pred_occupancy_next_week, pred_high_strain, pred_high_strain_prob,
        actual_occupancy, actual_high_strain, abs_error, shap_base_value,
        top1_feature, top1_shap, top1_direction,
        top2_feature, top2_shap, top2_direction,
        top3_feature, top3_shap, top3_direction, explanation
    )
    SELECT
        CAST(model_name               AS STRING),
        CAST(hospital_id              AS STRING),
        SAFE_CAST(report_date         AS DATE),
        CAST(run_date                 AS DATE),
        CAST(pred_occupancy_next_week AS FLOAT64),
        CAST(pred_high_strain         AS INT64),
        CAST(pred_high_strain_prob    AS FLOAT64),
        -- actual_* NULL at inference time — backfill via separate actuals pipeline
        CAST(actual_occupancy         AS FLOAT64),
        CAST(actual_high_strain       AS INT64),
        CAST(abs_error                AS FLOAT64),
        CAST(shap_base_value          AS FLOAT64),
        CAST(top1_feature  AS STRING), CAST(top1_shap  AS FLOAT64), CAST(top1_direction  AS STRING),
        CAST(top2_feature  AS STRING), CAST(top2_shap  AS FLOAT64), CAST(top2_direction  AS STRING),
        CAST(top3_feature  AS STRING), CAST(top3_shap  AS FLOAT64), CAST(top3_direction  AS STRING),
        CAST(explanation   AS STRING)
    FROM `{RAW_TABLE}`
    WHERE run_date = '{{{{ ds }}}}';
"""

_sql_forecast = f"""
    DELETE FROM `{FORECAST_TABLE}` WHERE run_date = '{{{{ ds }}}}';

    INSERT INTO `{FORECAST_TABLE}` (
        forecast_id, hospital_id, hospital_name, state, run_date, forecast_date,
        predicted_occupancy_rate, confidence_lower_95, confidence_upper_95,
        alert_flag, shap_feature_1, shap_feature_2, shap_feature_3,
        shap_value_1, shap_value_2, shap_value_3, model_version
    )
    SELECT
        GENERATE_UUID()                                                     AS forecast_id,
        p.hospital_id,
        d.hospital_name,
        d.state,
        p.run_date,
        -- forecast_date: next-week prediction target
        DATE_ADD(SAFE_CAST(p.report_date AS DATE), INTERVAL 7 DAY)         AS forecast_date,
        p.pred_occupancy_next_week                                          AS predicted_occupancy_rate,
        -- Confidence intervals: ±0.05 approximation.
        -- TODO: replace with model-native intervals (quantile regression / conformal prediction).
        GREATEST(0.0, p.pred_occupancy_next_week - 0.05)                   AS confidence_lower_95,
        LEAST(1.0,    p.pred_occupancy_next_week + 0.05)                   AS confidence_upper_95,
        -- alert_flag: high occupancy risk flag (spec §5.6 SFR-07)
        (p.pred_occupancy_next_week >= 0.90)                                AS alert_flag,
        -- SHAP rename: topX_feature → shap_feature_X for serving schema convention
        p.top1_feature  AS shap_feature_1,
        p.top2_feature  AS shap_feature_2,
        p.top3_feature  AS shap_feature_3,
        p.top1_shap     AS shap_value_1,
        p.top2_shap     AS shap_value_2,
        p.top3_shap     AS shap_value_3,
        p.model_version
    FROM `{RAW_TABLE}` p
    LEFT JOIN `{DIM_TABLE}` d ON p.hospital_id = d.hospital_id
    WHERE p.run_date   = '{{{{ ds }}}}'
      AND p.model_name = 'XGBoost';
"""


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="hospital_ml_inference",
    description="Weekly hospital occupancy batch inference → serving tables",
    schedule_interval="0 2 * * 1",   # Monday 02:00 UTC
    start_date=datetime(2026, 6, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["ml", "inference", "hospital", "weekly"],
    params={
        "lookback_days": Param(
            default=40,
            type="integer",
            description=(
                "Days to look back for feature data. "
                "Default 40 = covers max 5-week HHS update cycle. "
                "Increase for backfill runs."
            ),
        ),
    },
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # ── Guard: feature store freshness ────────────────────────────────────────
    validate_feature_store = BigQueryCheckOperator(
        task_id="validate_feature_store",
        sql=(
            "SELECT "
            "  CAST(COUNT(*) > 0 AS INT64)                                              AS has_data, "
            "  CAST(DATE_DIFF(DATE('{{ ds }}'), MAX(report_date), DAY) <= 40 AS INT64)  AS data_not_stale "
            f"FROM `{INFER_FS_TABLE}` "
            "WHERE report_date <= DATE('{{ ds }}')"
        ),
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md=(
            "Guard: feature store has data AND latest snapshot ≤ 40 days old. "
            "HHS updates every 3-4 weeks — 40-day ceiling = 5-week buffer. "
            "> 40 days = genuine pipeline failure (ingestion broken), not normal HHS latency."
        ),
    )

    # ── Guard: skip if no new HHS data since last run ─────────────────────────
    skip_if_no_new_data = ShortCircuitOperator(
        task_id="skip_if_no_new_data",
        python_callable=_has_new_feature_data,
        ignore_downstream_trigger_rules=True,  # Skip ALL downstream when False
        doc_md=(
            "Skips CF trigger when HHS has not published new data since last inference run. "
            "~75% of Mondays are no-ops. Saves CF compute cost + prevents duplicate run_date partitions."
        ),
    )

    # ── ML compute: trigger Cloud Function ───────────────────────────────────
    trigger_inference_cf = PythonOperator(
        task_id="trigger_inference_cf",
        python_callable=_trigger_inference_cf,
        execution_timeout=timedelta(minutes=30),  # CF max 60min + buffer
        retries=2,
        retry_delay=timedelta(minutes=5),
        doc_md=(
            "HTTP POST to CF inference worker (hospital-inference-pipeline). "
            "CF scope: load BQ features → XGB/LGB predict → SHAP → write raw table. "
            "Blocks synchronously until CF responds. CF 500 → Airflow retries this task only."
        ),
    )

    # ── Guard: raw predictions table populated ────────────────────────────────
    validate_raw_predictions = BigQueryCheckOperator(
        task_id="validate_raw_predictions",
        sql=(
            "SELECT "
            "  CAST(COUNT(*) >= 1 AS INT64)                                   AS has_rows, "
            "  CAST(COUNTIF(model_name = 'XGBoost') > 0 AS INT64)            AS has_xgboost, "
            "  CAST(COUNTIF(pred_occupancy_next_week IS NULL) = 0 AS INT64)   AS no_null_preds, "
            "  CAST(COUNTIF(top1_feature IS NULL) = 0 AS INT64)              AS no_null_shap "
            f"FROM `{RAW_TABLE}` "
            "WHERE run_date = '{{ ds }}'"
        ),
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md=(
            "Fail-fast guard before post-processing: raw table must have rows, "
            "XGBoost predictions, and no NULL SHAP values. "
            "If this fails, post-processing is blocked → serving tables retain last-run data."
        ),
    )

    # ── Post-process: raw → observability + forecast (business SQL) ───────────
    run_post_process = BigQueryInsertJobOperator(
        task_id="run_post_process",
        configuration={
            "query": {
                "query": f"{_sql_observability}\n{_sql_forecast}",
                "useLegacySql": False,
            }
        },
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md=(
            "Post-process raw predictions → serving tables. "
            "Business logic: alert_flag (>=90%), forecast_date (+7d), "
            "confidence intervals (±0.05), dim JOIN for hospital_name/state. "
            "SHAP column rename: topX_feature → shap_feature_X. "
            "Idempotent: DELETE WHERE run_date → INSERT. Safe to retry standalone."
        ),
    )

    # ── Guard: serving table validation ──────────────────────────────────────
    # <5% NULL hospital_name is acceptable — hospitals not in dim still appear via LEFT JOIN.
    validate_forecast_results = BigQueryCheckOperator(
        task_id="validate_forecast_results",
        sql=(
            "SELECT "
            "  CAST(COUNT(*) >= 1 AS INT64)                          AS has_rows, "
            "  CAST(COUNTIF(alert_flag IS NULL) = 0 AS INT64)        AS no_null_alert, "
            "  CAST(COUNTIF(shap_feature_1 IS NULL) = 0 AS INT64)    AS no_null_shap, "
            "  CAST(COUNTIF(hospital_name IS NULL) / NULLIF(COUNT(*), 0) < 0.05 AS INT64) "
            "                                         AS hospital_join_ok "
            f"FROM `{FORECAST_TABLE}` "
            "WHERE run_date = '{{ ds }}'"
        ),
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md="Validates ml_forecast_results: rows, alert_flag, SHAP, dim JOIN quality.",
    )

    validate_observability = BigQueryCheckOperator(
        task_id="validate_observability",
        sql=(
            "SELECT "
            "  CAST(COUNT(*) >= 1 AS INT64)                                AS has_rows, "
            "  CAST(COUNTIF(pred_occupancy_next_week IS NULL) = 0 AS INT64) AS no_null_preds, "
            "  CAST(COUNT(DISTINCT model_name) >= 1 AS INT64)              AS has_models "
            f"FROM `{OBSERVABILITY_TABLE}` "
            "WHERE run_date = '{{ ds }}'"
        ),
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md="Validates model_predictions_shap: rows, no NULL predictions, model names present.",
    )

    # ── DAG dependencies ──────────────────────────────────────────────────────
    (
        start
        >> validate_feature_store
        >> skip_if_no_new_data
        >> trigger_inference_cf
        >> validate_raw_predictions
        >> run_post_process
        >> [validate_forecast_results, validate_observability]
        >> end
    )