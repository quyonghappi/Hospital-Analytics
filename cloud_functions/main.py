import logging
import sys
from datetime import datetime, timezone

import functions_framework
import pandas as pd
from google.cloud import bigquery, storage

from modules.config import (
    BUCKET_NAME,
    INFER_FS_TABLE,
    LGB_PREFIX,
    PROJECT_ID,
    RAW_TABLE,
    XGB_PREFIX,
)
from modules.inference import (
    load_feature_data,
    load_model_artifacts,
    log_inference_qa,
    resolve_feature_schema,
    run_inference,
    upload_raw_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("cf-inference-worker")


# ─── Request Parsing & Validation ────────────────────────────────────────────

def _parse_request(request) -> dict:
    """
    Parse HTTP payload and apply defaults + validation.

    Strict validation here prevents bad input from propagating deep into
    BQ queries or GCS lookups where the error message is harder to interpret.

    run_date default: UTC today — CF instances may run in non-UTC local timezones,
    so date.today() is NOT safe here. Always derive from timezone.utc.
    """
    body = request.get_json(silent=True) or {}

    run_date_raw = body.get(
        "run_date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    try:
        datetime.strptime(run_date_raw, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid run_date: '{run_date_raw}'. Expected YYYY-MM-DD."
        )

    write_disposition = body.get("write_disposition", "WRITE_APPEND")
    if write_disposition not in {"WRITE_APPEND", "WRITE_TRUNCATE"}:
        raise ValueError(
            f"Invalid write_disposition: '{write_disposition}'. "
            "Must be WRITE_APPEND or WRITE_TRUNCATE."
        )

    lookback_days = int(body.get("lookback_days", 40))
    if lookback_days < 1 or lookback_days > 365:
        raise ValueError(
            f"lookback_days={lookback_days} out of range [1, 365]."
        )

    return {
        "run_date":          run_date_raw,
        "lookback_days":     lookback_days,
        "write_disposition": write_disposition,
        "skip_lgb":          bool(body.get("skip_lgb", False)),
    }


@functions_framework.http
def inference_entrypoint(request):
    start_time = datetime.now(timezone.utc)

    # Separated from pipeline try/catch so run_date is always defined in except branches.
    try:
        params = _parse_request(request)
    except ValueError as e:
        logger.error("[Request] Invalid payload: %s", e)
        return {"status": "error", "error_type": "BadRequest", "message": str(e)}, 400

    run_date          = params["run_date"]
    lookback_days     = params["lookback_days"]
    write_disposition = params["write_disposition"]
    dry_run           = params["dry_run"]
    skip_lgb          = params["skip_lgb"]

    logger.info("=" * 60)
    logger.info("CF Inference Worker START")
    logger.info("  run_date          : %s", run_date)
    logger.info("  lookback_days     : %d", lookback_days)
    logger.info("  write_disposition : %s", write_disposition)
    logger.info("  dry_run           : %s", dry_run)
    logger.info("  skip_lgb          : %s", skip_lgb)
    logger.info("=" * 60)

    try:
        bq_client  = bigquery.Client(project=PROJECT_ID)
        gcs_client = storage.Client(project=PROJECT_ID)

        logger.info("[Phase 1] Loading feature data from BigQuery...")
        df = load_feature_data(
            bq_client, INFER_FS_TABLE, run_date, lookback_days
        )

        logger.info("[Phase 2] Loading XGBoost artifacts from GCS...")
        xgb_artifacts = load_model_artifacts(
            gcs_client, BUCKET_NAME, XGB_PREFIX, model_label="XGBoost"
        )
        feature_cols, num_feats, cat_feats = resolve_feature_schema(
            xgb_artifacts["preprocessor"],
            df,
            xgb_artifacts.get("feature_metadata"),
        )
        all_feat_names = num_feats + cat_feats

        logger.info("[Phase 3] Running XGBoost inference...")
        all_records = run_inference(
            artifacts=xgb_artifacts,
            df=df,
            feature_cols=feature_cols,
            all_feat_names=all_feat_names,
            model_name="XGBoost",
            run_date=run_date,
            source_table=INFER_FS_TABLE,
        )

        if skip_lgb:
            logger.info("[Phase 4] LightGBM skipped (skip_lgb=True).")
        elif not LGB_PREFIX:
            logger.info("[Phase 4] LightGBM skipped (LGB_PREFIX not configured).")
        else:
            logger.info("[Phase 4] Running LightGBM inference (optional)...")
            try:
                lgb_artifacts = load_model_artifacts(
                    gcs_client, BUCKET_NAME, LGB_PREFIX, model_label="LightGBM"
                )
                # Reuse same feature_cols / all_feat_names — shared preprocessor
                lgb_records = run_inference(
                    artifacts=lgb_artifacts,
                    df=df,
                    feature_cols=feature_cols,
                    all_feat_names=all_feat_names,
                    model_name="LightGBM",
                    run_date=run_date,
                    source_table=INFER_FS_TABLE,
                )
                all_records = pd.concat([all_records, lgb_records], ignore_index=True)
                logger.info(
                    "[Phase 4] LightGBM complete: %d rows appended.",
                    len(lgb_records),
                )
            except FileNotFoundError as e:
                # Non-blocking — LightGBM is secondary model, don't fail the pipeline
                logger.warning(
                    "[Phase 4] LightGBM artifacts not found — skipping (non-blocking): %s", e
                )

        log_inference_qa(all_records)

        # ── Phase 5: Upload raw predictions ──────────────────────────────────
        # Writes ONLY to vertex_batch_prediction_raw (staging table).
        # Idempotent: DELETE WHERE run_date=X → INSERT.
        # Airflow validate_raw_predictions checks this table before post-processing runs.
        # Airflow handles all downstream serving table writes (observability, forecast).
        logger.info("[Phase 5] Uploading raw predictions to staging table...")
        if not dry_run:
            upload_raw_predictions(
                bq_client=bq_client,
                df=all_records,
                raw_table=RAW_TABLE,
                run_date=run_date,
                write_disposition=write_disposition,
            )
        else:
            logger.info(
                "[Phase 5] DRY RUN — %d rows would be written to %s.",
                len(all_records), RAW_TABLE,
            )

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed = _elapsed(start_time)
        records_by_model: dict = all_records.groupby("model_name").size().to_dict()

        logger.info("=" * 60)
        logger.info(
            "CF Inference Worker COMPLETE | %.1fs | run_date=%s | records=%s",
            elapsed, run_date, records_by_model,
        )
        logger.info("  → Airflow will validate raw table and run post-processing.")
        logger.info("=" * 60)

        return {
            "status":         "success",
            "run_date":       run_date,
            "elapsed_sec":    elapsed,
            "records":        records_by_model,
            "dry_run":        dry_run,
            "raw_table":      RAW_TABLE,
        }, 200

    # ── Classified error responses ────────────────────────────────────────────
    # Returning a specific error_type lets Cloud Monitoring / alerting policies
    # distinguish between data outages (DataValidationError) and infra failures
    # (ArtifactNotFoundError, SchemaMismatchError) without parsing message strings.
    # HTTP 500 causes Airflow to mark trigger_inference_cf as FAILED → retry.

    except ValueError as e:
        # Raised by: load_feature_data (empty BQ result)
        return _error_response("DataValidationError", e, run_date, start_time)

    except FileNotFoundError as e:
        # Raised by: load_model_artifacts (mandatory artifact blobs missing)
        return _error_response("ArtifactNotFoundError", e, run_date, start_time)

    except AssertionError as e:
        # Raised by: run_inference (X.shape[1] != len(all_feat_names) — schema mismatch)
        return _error_response("SchemaMismatchError", e, run_date, start_time)

    except Exception as e:
        # Catch-all: BQ API errors, GCS network issues, unexpected bugs
        return _error_response(type(e).__name__, e, run_date, start_time)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _elapsed(start: datetime) -> float:
    """Seconds elapsed since start (UTC)."""
    return round((datetime.now(timezone.utc) - start).total_seconds(), 1)


def _error_response(
    error_type: str,
    exc: Exception,
    run_date: str,
    start_time: datetime,
) -> tuple[dict, int]:
    """
    Uniform 500 error response.

    All 500s log exc_info=True so Cloud Logging captures the full traceback.
    HTTP 500 → Airflow marks trigger_inference_cf FAILED → retries per task retry config.
    """
    elapsed = _elapsed(start_time)
    logger.error(
        "[Pipeline] %s after %.1fs — run_date=%s: %s",
        error_type, elapsed, run_date, exc,
        exc_info=True,
    )
    return {
        "status":      "error",
        "error_type":  error_type,
        "message":     str(exc),
        "run_date":    run_date,
        "elapsed_sec": elapsed,
    }, 500