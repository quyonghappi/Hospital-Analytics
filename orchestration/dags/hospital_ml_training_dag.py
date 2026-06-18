from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
# from airflow.providers.google.cloud.operators.vertex_ai.custom_job import CreateCustomContainerTrainingJobOperator
from operators import CreateCustomContainerTrainingJobOperator
from airflow.utils.trigger_rule import TriggerRule

from config import (
    PROJECT_ID,
    REGION,
    BUCKET_NAME,
    IMAGE_URI,
    STAGING_GCS,
    TRAIN_FS_TABLE,
    XGB_PREFIX,
    LGB_PREFIX,
    TRAIN_END,
    VAL_END,
    MACHINE_TRAIN_PIPELINE,
    GCP_CONN_ID,
    SERVICE_ACCOUNT,
)

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Artifact dir bên trong Vertex AI container — shared bởi data_prep, train, evaluate.
# Phải match với --artifact_dir default trong cả 3 scripts.
_ARTIFACT_DIR = "/app/ml_artifacts"

# Minimum rows tối thiểu trong feature store trước khi cho phép training.
# ~500 hospitals × 104 tuần (2 năm) ≈ 52k rows baseline.
# < 10k rows = data pipeline failure, không phải model issue.
_MIN_TRAINING_ROWS = 10_000

# ─── Command Template ─────────────────────────────────────────────────────────
#
# Chained bash command chạy trong 1 container:
#   Step 1: run_train_pipeline.py  (data_prep → train → upload artifacts to GCS)
#   Step 2: evaluate_and_register.py (test eval → gate → Vertex AI Model Registry)
#
# Kết hợp f-string (resolved tại DAG parse time) với Jinja2 {{ }} (resolved tại execution time).
# set -e: đảm bảo bash exit ngay khi step 1 fail — không run evaluate nếu training fail.
#
# NOTE: nếu cần tách evaluate thành separate Vertex AI job trong tương lai,
# run_train_pipeline.py phải upload test_features.parquet lên GCS, và
# evaluate_and_register.py cần thêm GCS fallback cho test_features.parquet.

_TRAIN_AND_EVAL_CMD = (
    "set -e"
    # ── Step 1: data_prep + train ──────────────────────────────────────────────
    " && python run_train_pipeline.py"
    f" --project_id      {PROJECT_ID}"
    f" --bucket_name     {BUCKET_NAME}"
    f" --artifact_dir    {_ARTIFACT_DIR}"
    f" --xgb_prefix      {XGB_PREFIX}"
    f" --lgb_prefix      {LGB_PREFIX}"
    # Jinja2 params
    " --train_end        {{ params.train_end }}"
    " --val_end          {{ params.val_end }}"
    " --n_estimators     {{ params.n_estimators | int }}"
    # ── Step 2: evaluate_and_register ─────────────────────────────────────────
    " && python src/evaluate_and_register.py"
    f" --project_id      {PROJECT_ID}"
    f" --region          {REGION}"
    f" --bucket_name     {BUCKET_NAME}"
    f" --artifact_dir    {_ARTIFACT_DIR}"
    f" --xgb_prefix      {XGB_PREFIX}"
    f" --lgb_prefix      {LGB_PREFIX}"
    " --mae_threshold        {{ params.mae_threshold }}"
    " --r2_threshold         {{ params.r2_threshold }}"
    " --roc_auc_threshold    {{ params.roc_auc_threshold }}"
    # Conditional flags: rendered by Jinja2, produce empty string if False
    "{% if params.skip_lgb %} --skip_lgb{% endif %}"
    "{% if params.dry_run %} --dry_run{% endif %}"
)

DEFAULT_ARGS = {
    "owner":             "ml-team",
    "depends_on_past":   False,
    # retries=0: training failure requires human review, không nên auto-retry.
    "retries":           0,
    # 6h ceiling: data_prep ~15min + train ~30min + eval ~10min = ~55min typical.
    # n1-highmem-8 với 500 hospitals
    "execution_timeout": timedelta(hours=6),
    # TODO: thêm on_failure_callback khi Slack/PagerDuty infra sẵn sàng
    # "on_failure_callback": slack_alert,
}

# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="hospital_ml_training",
    description="Hospital occupancy model retraining: data_prep → train → evaluate_and_register",
    schedule_interval=None,          # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,               # Prevent concurrent training runs on same GCS prefix
    tags=["ml", "training", "hospital", "manual"],
    params={
        # Override tại runtime qua Airflow UI → Trigger DAG w/ config
        "train_end": Param(
            default=TRAIN_END,
            type="string",
            description="Train split end date (YYYY-MM-DD). Tất cả data ≤ date này = training set.",
        ),
        "val_end": Param(
            default=VAL_END,
            type="string",
            description="Val split end date (YYYY-MM-DD). Data giữa train_end và val_end = val set.",
        ),
        "n_estimators": Param(
            default=300,
            type="integer",
            description="Boosting rounds cho XGBoost/LightGBM. Notebook baseline: 300.",
        ),
        "skip_lgb": Param(
            default=False,
            type="boolean",
            description="Bỏ qua LightGBM training + evaluation. Dùng khi chỉ cần retrain XGBoost nhanh.",
        ),
        "dry_run": Param(
            default=False,
            type="boolean",
            description="Run evaluation + gate nhưng KHÔNG register vào Vertex AI Model Registry.",
        ),
        # Gate thresholds — override nếu muốn relax/tighten gate trong 1 training run cụ thể
        "mae_threshold": Param(
            default=0.10,
            type="number",
            description="Max test MAE để pass gate (default: 0.10 = 10% absolute error trên range 0-1).",
        ),
        "r2_threshold": Param(
            default=0.75,
            type="number",
            description="Min test R2 để pass gate (default: 0.75). Notebook baseline: 0.82-0.83.",
        ),
        "roc_auc_threshold": Param(
            default=0.85,
            type="number",
            description="Min test ROC-AUC để pass gate (default: 0.85). Notebook baseline: 0.94-0.95.",
        ),
    },
    doc_md="""
## Hospital ML Training Pipeline

Manual retraining pipeline — chạy theo yêu cầu, không scheduled.

**Source:** `hospital_feature_store.fs_hospital_weekly`
**Model artifacts → GCS:** `hospital-model/xgboost/` + `hospital-model/lightgbm/`
**Model Registry:** Vertex AI Model Registry (`hospital-occupancy-xgboost`, `hospital-occupancy-lightgbm`)

### Trigger scenarios
- **Quarterly**: 3 tháng HHS data mới → update train_end/val_end
- **Model drift**: MLOps dashboard shows MAE trend > 0.02 sustained over 4+ weeks
- **Feature update**: fs_hospital_weekly schema thay đổi → xác nhận feature_metadata.json updated

### Gate outcomes
| Exit code | Meaning | Airflow task | Inference DAG |
|---|---|---|---|
| 0 | All passing models registered | SUCCESS | Unchanged (reads Model Registry) |
| 2 | Gate fail — models dưới ngưỡng | FAILED | Unchanged — production model retained |
| 1 | Unexpected crash | FAILED | Unchanged |

### Important
`notify_training_result` luôn chạy (ALL_DONE) để capture outcome trong logs.
Inference DAG **không** auto-trigger sau training — decouple by design.
Model Registry `status=production` là source of truth cho inference.
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # notify_training_result: luôn chạy bất kể gate pass/fail/crash
    # Đây là hook point cho alert callback (Slack/PagerDuty)
    # Khi gate fail: log WARN + alert on-call. Khi success: log INFO + notify team.
    notify_training_result = EmptyOperator(
        task_id="notify_training_result",
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md=(
            "Runs regardless of training outcome (ALL_DONE). "
            "Replace EmptyOperator với PythonOperator + Slack/PagerDuty callback "
            "khi alerting infra sẵn sàng. "
            "Check XCom của run_train_and_evaluate để biết gate outcome."
        ),
    )

    # end: ALL_DONE để DAG không stuck khi run_train_and_evaluate fail
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── 1. Guard: feature store có đủ history để train ────────────────────────
    #
    # Fail fast trước khi submit Vertex AI job (tiết kiệm ~60-90min + n1-highmem-8 cost).
    # Condition: COUNT(rows) >= _MIN_TRAINING_ROWS với report_date <= train_end param.
    #
    # KHÔNG check report_date > val_end ở đây — test set size được validate trong
    # data_prep.py perform_temporal_split() với --min_rows_per_split guard.
    #
    # BigQueryCheckOperator returns TRUE nếu first row, first column là truthy.
    # COUNT(*) >= 10000 → returns 1 (TRUE) nếu pass, 0 (FALSE) nếu fail.
    validate_training_data_size = BigQueryCheckOperator(
        task_id="validate_training_data_size",
        sql=(
            f"SELECT CAST(COUNT(*) >= {_MIN_TRAINING_ROWS} AS INT64) "
            f"FROM `{TRAIN_FS_TABLE}` "
            "WHERE report_date <= '{{ params.train_end }}'"
        ),
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID,
        location=REGION,
        doc_md=(
            f"Guard: feature store phải có ≥ {_MIN_TRAINING_ROWS:,} rows tính đến train_end. "
            f"Fail = data pipeline chưa chạy đủ. Không phải model issue."
        ),
    )

    # ── 2. Training + Evaluation — Single Vertex AI Custom Job ────────────────
    #
    # Machine: n1-highmem-8 (52 GB RAM, 8 vCPU)
    #   - Dominated bởi data_prep memory requirement (pandas DataFrame full feature store)
    #   - train step (XGBoost + LightGBM + SHAP): ~8-16 GB
    #   - evaluate step: < 8 GB
    #   - Tất cả chạy tuần tự trong 1 container, peak memory ≈ data_prep peak
    #
    # Entrypoint override: /bin/bash -c
    #   Cho phép chaining với && và conditional flags (--skip_lgb, --dry_run)
    #   Container default ENTRYPOINT (python) bị override
    #
    # worker_pool_specs là template_field trong CreateCustomJobOperator
    # → Jinja2 {{ params.* }} và {% if %} trong args strings được rendered tại execution time
    run_train_and_evaluate = CreateCustomContainerTrainingJobOperator(
        task_id="run_train_and_evaluate",
        project_id=PROJECT_ID,
        region=REGION,
        display_name="hospital-training-{{ ds_nodash }}-te{{ params.train_end | replace('-', '') }}",
        container_uri=IMAGE_URI,           # ← REQUIRED, was buried in container_spec
        command=["/bin/bash", "-c"],        # ← lifted from container_spec
        args=[_TRAIN_AND_EVAL_CMD],         # ← lifted from container_spec
        machine_type=MACHINE_TRAIN_PIPELINE,  # ← lifted from machine_spec
        replica_count=1,
        service_account=SERVICE_ACCOUNT,
        staging_bucket=STAGING_GCS,
        gcp_conn_id=GCP_CONN_ID,
    )

    (
        start
        >> validate_training_data_size
        >> run_train_and_evaluate
        >> notify_training_result
        >> end
    )