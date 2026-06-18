import logging
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from google import genai
from google.genai import types
from google.genai.types import FinishReason

import pandas as pd
import yaml
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted, ServiceUnavailable
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger("chatbot_engine")

_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|TRUNCATE|MERGE|ALTER|REPLACE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_SQL_FENCE_PATTERN = re.compile(
    r"```(?:sql)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

_ANOMALY_PCT_THRESHOLD = 200.0   # Any pct column > 200 → flag as pipeline anomaly
_PCT_COLUMNS = frozenset({       # Columns to check for anomalous values
    "occupancy_pct", "icu_occupancy_pct", "predicted_pct",
    "avg_occupancy_pct", "avg_icu_occupancy_pct",
    "conf_lower_pct", "conf_upper_pct",
})


class ErrorCode(str, Enum):
    """
    Canonical status taxonomy for all chatbot outcomes.

    Used as `status` field in the API response dict.
    str mixin ensures JSON-serialisable without extra conversion.
    """
    SUCCESS       = "success"        # Query ran, rows returned, NL formatted
    NO_DATA       = "no_data"        # Query valid, BQ returned zero rows
    CANNOT_ANSWER = "cannot_answer"  # Question outside analytics scope
    SQL_BLOCKED   = "sql_blocked"    # DML / security violation (never expose details)
    TIMEOUT       = "timeout"        # BQ query exceeded 20s server-side timeout
    RATE_LIMITED  = "rate_limited"   # Gemini quota exhausted after all retries
    ERROR         = "error"          # Unexpected internal error


# User-facing messages
_USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.NO_DATA: (
        "No data found for your query. "
        "This may mean no hospitals meet the specified criteria, "
        "or the data for the requested period is not yet available."
    ),
    ErrorCode.CANNOT_ANSWER: (
        "I can only answer questions about hospital capacity, occupancy, and forecasts. "
        "Try asking about occupancy rates, available beds, overload risk, "
        "ICU trends, or regional benchmarks."
    ),
    ErrorCode.SQL_BLOCKED: (
        "I can't process that request. "
        "Please ask about hospital capacity, occupancy rates, or forecast data."
    ),
    ErrorCode.TIMEOUT: (
        "The query took too long to complete. "
        "Try being more specific — add a hospital name, state, or county to narrow results."
    ),
    ErrorCode.RATE_LIMITED: (
        "The service is temporarily busy. Please try again in a moment."
    ),
    ErrorCode.ERROR: (
        "I encountered an error processing your request. "
        "Please try rephrasing your question, or contact the platform team "
        "if the issue persists."
    ),
}

_SCOPE_KEYWORDS = frozenset({
    # Entity types
    "hospital", "hospitals", "facility", "facilities", "clinic", "clinics",
    "medical center", "healthcare", "health system",
    # Capacity / beds
    "occupancy", "capacity", "bed", "beds", "icu", "inpatient",
    "staffed", "available", "overload", "overcapacity",
    # Operations
    "patient", "patients", "admission", "admissions", "staffing", "shortage",
    # Forecast / ML
    "forecast", "predict", "predicted", "prediction", "risk", "alert",
    "shap", "factor", "driver", "confidence",
    # Status labels
    "critical", "warning", "normal", "status", "threshold", "flag", "flagged",
    # Temporal / trend
    "trend", "trends", "week", "weekly", "month", "rolling",
    # Benchmark / geo
    "benchmark", "rank", "ranked", "ranking", "compare", "comparison",
    "hrr", "county", "region", "state", "referral",
    # Analytics qualifiers
    "above", "below", "exceed", "exceeds", "highest", "lowest", "most", "least",
    "overloaded", "surge", "utilization", "utilisation",
    # Disease-specific (present in actual schema)
    "covid", "influenza", "flu", "emergency", "ed",
    "pediatric", "pediatrics",
})

_GEMINI_RETRY_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0

_MAX_RESULT_CHARS = 10_000
_MAX_DISPLAY_ROWS = 100

_FEW_SHOT_EXAMPLES = [
    {
        "question": "Which hospitals in California are above 90% capacity right now?",
        "sql": """\
SELECT
  d.hospital_name,
  f.state,
  ROUND(f.inpatient_occupancy_rate, 1)  AS occupancy_pct,
  f.inpatient_beds_used,
  f.inpatient_beds_capacity,
  f.collection_week,
  CASE
    WHEN f.inpatient_occupancy_rate >= 0.9 THEN 'CRITICAL'
    WHEN f.inpatient_occupancy_rate >= 0.7 THEN 'WARNING'
    ELSE 'NORMAL'
  END AS status
FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization` f
JOIN `{project_id}.hospital_dwh_dev.dim_hospital` d
  ON f.hospital_id = d.hospital_id
WHERE f.collection_week = (
    SELECT MAX(collection_week)
    FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization`
  )
  AND f.inpatient_occupancy_rate > 0.9
  AND f.state = 'CA'
ORDER BY f.inpatient_occupancy_rate DESC
LIMIT 100""",
    },
    {
        # UC-02: Overload risk forecast summary
        "question": "Which hospitals are predicted to exceed 90% occupancy in the next 7 days?",
        "sql": """\
SELECT
  f.hospital_name,
  f.state,
  ROUND(f.predicted_occupancy_rate, 1)  AS predicted_pct,
  ROUND(f.confidence_lower_95, 1)       AS conf_lower_pct,
  ROUND(f.confidence_upper_95, 1)       AS conf_upper_pct,
  f.forecast_date,
  f.run_date
FROM `{project_id}.ml_predictions_dev.ml_forecast_results` f
WHERE f.run_date = (
    SELECT MAX(run_date)
    FROM `{project_id}.ml_predictions_dev.ml_forecast_results`
  )
  AND f.alert_flag = TRUE
ORDER BY f.predicted_occupancy_rate DESC
LIMIT 100""",
    },
    {
        # UC-03: SHAP-driven alert explanation
        "question": "Why is Valley Medical Center flagged as high risk next week?",
        "sql": """\
SELECT
  f.hospital_name,
  f.state,
  ROUND(f.predicted_occupancy_rate, 1)  AS predicted_pct,
  ROUND(f.confidence_lower_95, 1)       AS conf_lower_pct,
  ROUND(f.confidence_upper_95, 1)       AS conf_upper_pct,
  f.shap_feature_1,
  ROUND(f.shap_value_1, 4)                    AS shap_value_1,
  f.shap_feature_2,
  ROUND(f.shap_value_2, 4)                    AS shap_value_2,
  f.shap_feature_3,
  ROUND(f.shap_value_3, 4)                    AS shap_value_3,
  f.forecast_date,
  f.run_date,
  f.model_version
FROM `{project_id}.ml_predictions_dev.ml_forecast_results` f
WHERE f.run_date = (
    SELECT MAX(run_date)
    FROM `{project_id}.ml_predictions_dev.ml_forecast_results`
  )
  AND LOWER(f.hospital_name) LIKE LOWER('%valley medical%')
ORDER BY f.predicted_occupancy_rate DESC
LIMIT 10""",
    },
    {
        # UC-04: Capacity gap
        "question": "Which hospitals have the most available capacity to absorb more patients?",
        "sql": """\
SELECT
  d.hospital_name,
  f.state,
  g.county_name,
  f.inpatient_beds_capacity - f.inpatient_beds_used  AS available_beds,
  ROUND(f.inpatient_occupancy_rate, 1)         AS occupancy_pct,
  f.inpatient_beds_capacity,
  f.inpatient_beds_used,
  CASE
    WHEN f.inpatient_occupancy_rate >= 0.9 THEN 'CRITICAL'
    WHEN f.inpatient_occupancy_rate >= 0.7 THEN 'WARNING'
    ELSE 'NORMAL'
  END                                                 AS status,
  f.collection_week
FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization` f
JOIN `{project_id}.hospital_dwh_dev.dim_hospital` d
  ON f.hospital_id = d.hospital_id
JOIN `{project_id}.hospital_dwh_dev.dim_geography` g
  ON d.county_fips = g.county_fips
WHERE f.collection_week = (
    SELECT MAX(collection_week)
    FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization`
  )
  AND f.inpatient_occupancy_rate < 0.7
ORDER BY available_beds DESC
LIMIT 100""",
    },
    {
        # UC-05: 4-week rolling trend (ICU + inpatient)
        "question": "Show me the ICU occupancy trend over the past 4 weeks.",
        "sql": """\
SELECT
  d.hospital_name,
  f.state,
  DATE_TRUNC(f.collection_week, WEEK)                    AS week_start,
  ROUND(AVG(f.inpatient_occupancy_rate), 1)        AS avg_occupancy_pct,
  ROUND(AVG(f.icu_occupancy_rate), 1)              AS avg_icu_occupancy_pct,
  SUM(f.inpatient_beds_used)                             AS total_beds_in_use,
  SUM(f.covid_admissions_adult + f.covid_admissions_pediatric)  AS total_admissions
FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization` f
JOIN `{project_id}.hospital_dwh_dev.dim_hospital` d
  ON f.hospital_id = d.hospital_id
WHERE f.collection_week >= DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY)
GROUP BY
  d.hospital_name,
  f.state,
  week_start
ORDER BY
  d.hospital_name,
  week_start
LIMIT 100""",
    },
    {
        # UC-06: HRR benchmarking with CTE + RANK()
        "question": "Rank hospitals by occupancy rate within each HRR region this week.",
        "sql": """\
WITH current_week AS (
  SELECT
    f.hospital_id,
    f.state,
    f.inpatient_occupancy_rate,
    f.inpatient_beds_capacity,
    f.inpatient_beds_used,
    f.inpatient_beds_capacity - f.inpatient_beds_used  AS available_beds,
    f.collection_week
  FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization` f
  WHERE f.collection_week = (
      SELECT MAX(collection_week)
      FROM `{project_id}.hospital_dwh_dev.fact_hospital_utilization`
    )
)
SELECT
  d.hospital_name,
  cw.state,
  g.hrr_region,
  g.county_name,
  ROUND(cw.inpatient_occupancy_rate, 1)  AS occupancy_pct,
  cw.available_beds,
  cw.inpatient_beds_capacity,
  CASE
    WHEN cw.inpatient_occupancy_rate >= 0.9 THEN 'CRITICAL'
    WHEN cw.inpatient_occupancy_rate >= 0.7 THEN 'WARNING'
    ELSE 'NORMAL'
  END                                           AS status,
  RANK() OVER (
    PARTITION BY g.hrr_region
    ORDER BY cw.inpatient_occupancy_rate DESC
  )                                             AS rank_in_hrr,
  cw.collection_week
FROM current_week cw
JOIN `{project_id}.hospital_dwh_dev.dim_hospital` d
  ON cw.hospital_id = d.hospital_id
JOIN `{project_id}.hospital_dwh_dev.dim_geography` g
  ON d.county_fips = g.county_fips
ORDER BY g.hrr_region, rank_in_hrr
LIMIT 100""",
    },
]

class ChatbotEngine:
    """
    Text-to-SQL orchestrator: NL → SQL → BQ → NL.

    Thread-safe: BQ client and Vertex AI models are stateless per call.
    State: only the pre-built system prompt string (_sql_system_prompt).

    Error handling contract (chat() always returns, never raises):
      status=success       → data returned and NL-formatted
      status=no_data       → query valid, zero rows from BQ
      status=cannot_answer → out of scope (scope pre-filter or SQL guard stage 1/3)
      status=sql_blocked   → DML injection blocked (SQL field cleared in response)
      status=timeout       → BQ DeadlineExceeded; advise user to narrow query
      status=rate_limited  → Gemini ResourceExhausted after _MAX_RETRIES retries
      status=error         → unexpected exception; logged with exc_info

    Note on multi-turn: each call is stateless. The LLM has no memory of
    prior turns. Questions must be fully self-contained.
    """
    def __init__(
        self,
        project_id: str,
        region: str,
        schema_path: str,
        audit_table: str,
        gemini_model: str = "gemini-3.5-flash",
    ):
        
        self.project_id  = project_id
        self.region      = region
        self.audit_table = audit_table
        self.gemini_model = gemini_model

        self.bq_client = bigquery.Client(project=project_id)

        self.client = genai.Client(
            enterprise=True,
            project=project_id,
            location=region 
        )

        with open(schema_path) as f:
            self._schema = yaml.safe_load(f)

        self._schema_project_id = self._schema.get("project_id", project_id)

        self._sql_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=3000,
            candidate_count=1,
            system_instruction="",
            stop_sequences=["</SQL>", "Q:", "---"],
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )

        self._nl_config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
            candidate_count=1,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )

        self._sql_system_prompt = self._build_sql_system_prompt()

        self._sql_config.system_instruction = self._sql_system_prompt

        estimated_tokens = len(self._sql_system_prompt) // 4  # rough 4 chars/token heuristic
        logger.info(
            "System prompt stats | chars=%d | estimated_tokens≈%d | "
            "WARNING: if estimated_tokens > 4000, review max_output_tokens budget",
            len(self._sql_system_prompt),
            estimated_tokens,
        )

        if estimated_tokens > 6000:
            logger.warning(
                "LARGE SYSTEM PROMPT detected (~%d tokens). "
                "Consider compressing schema or splitting few-shots. "
                "Current max_output_tokens=%d may be insufficient.",
                estimated_tokens,
                self._sql_config.max_output_tokens,
            )

        logger.info(
            "ChatbotEngine initialized | project=%s | location=%s | model=%s | "
            "schema_tables=%d | prompt_chars=%d",
            project_id,
            region,
            gemini_model,
            len(self._schema.get("tables", {})),
            len(self._sql_system_prompt),
        )

    # ─── System Prompt Construction ───────────────────────────────────────────

    def _strip_sql_comments(self, sql: str) -> str:
        """Remove single-line SQL comments to reduce token count."""
        return re.sub(r"--[^\n]*", "", sql).strip()

    def _build_sql_system_prompt(self) -> str:
        pid = self._schema_project_id
        
        table_docs = []
        for _, info in self._schema.get("tables", {}).items():
            col_lines = "\n".join(
                f"    {col}: {desc}"
                for col, desc in info.get("columns", {}).items()
                if not col.startswith("_")  # skip _omit_from_prompt and similar
            )
            table_docs.append(
                f"  {info.get('full_path', 'MISSING_FULL_PATH')}\n"
                f"  -- {info.get('description', 'No description provided')}\n"
                f"  -- Granularity: {info.get('granularity', 'N/A')}\n"
                f"  Columns:\n{col_lines}"
            )
        tables_block = "\n\n".join(table_docs)

        # ── Glossary block ────────────────────────────────────────────────
        glossary_block = "\n".join(
            f"  {term}: {defn}"
            for term, defn in self._schema.get("business_glossary", {}).items()
        )

        # ── Constraints block ─────────────────────────────────────────────
        constraints_block = "\n".join(
            f"  - {c}" for c in self._schema.get("sql_constraints", [])
        )

        # ── Few-shots block ───────────────────────────────────────────────
        shot_parts = []
        for ex in _FEW_SHOT_EXAMPLES:
            sql = ex["sql"].replace("{project_id}", pid)
            sql = self._strip_sql_comments(sql)
            # shot_parts.append(f"Q: {ex['question']}\nSQL:\n```sql\n{sql}\n```")
            shot_parts.append(f"Q: {ex['question']}\nSQL:\n{sql}\n</SQL>")
        shots_block = "\n\n".join(shot_parts)

        return (
            "You are a BigQuery SQL analyst for a hospital capacity management platform.\n"
            "Given a natural language question, return a single valid BigQuery Standard SQL SELECT query.\n\n"
            "=== AVAILABLE TABLES ===\n"
            f"{tables_block}\n\n"
            "=== BUSINESS GLOSSARY ===\n"
            f"{glossary_block}\n\n"
            "=== SQL RULES (ALL MANDATORY) ===\n"
            f"{constraints_block}\n\n"
            "=== FEW-SHOT EXAMPLES ===\n"
            f"{shots_block}\n\n"
            "Return ONLY the SQL query. Do NOT wrap it in Markdown blockquotes (```). End your query with </SQL>."
        )

    def _is_in_scope(self, user_query: str) -> bool:
        q = user_query.lower()
        return any(kw in q for kw in _SCOPE_KEYWORDS)


    def _generate_sql_with_retry(self, user_query: str) -> str:
        full_prompt = f"Q: {user_query}\nSQL:\n"
        # full_prompt = f"Q: {user_query}\nSQL:\n```sql\n"
        # full_prompt = f"{self._sql_system_prompt}\n\nExample 7 — {user_query}\n"
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=full_prompt,
                    config=self._sql_config
                )
                raw_text = response.text or ""
    
                # ── Explicit finish reason check ──────────────────────────────
                finish_reason = None
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason

                if finish_reason == FinishReason.MAX_TOKENS:
                    # Truncated output — never usable as SQL
                    logger.warning(
                        "[gemini-retry] Attempt %d/%d: MAX_TOKENS hit. "
                        "Output chars=%d. Prompt may be too large or max_output_tokens too small.",
                        attempt + 1, _MAX_RETRIES + 1, len(raw_text),
                    )
                    # Treat as retryable — model was generating valid content
                    last_exc = ValueError(f"MAX_TOKENS truncation on attempt {attempt + 1}")
                    if attempt < _MAX_RETRIES:
                        time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue

                if finish_reason not in (FinishReason.STOP, None):
                    # SAFETY, RECITATION, OTHER — non-retryable
                    logger.warning(
                        "[gemini-retry] Non-retryable finish reason: %s", finish_reason
                    )
                    raise ValueError(f"Model stopped with reason: {finish_reason}")

                if not raw_text.strip():
                    raise ValueError(f"Empty response from model (Reason: {finish_reason})")

                return raw_text.strip()
            except _GEMINI_RETRY_EXCEPTIONS as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "[gemini-retry] Attempt %d/%d failed (%s). Retrying in %.1fs.",
                        attempt + 1, _MAX_RETRIES + 1, type(e).__name__, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[gemini-retry] All %d attempts exhausted: %s",
                        _MAX_RETRIES + 1, e,
                    )

        raise ResourceExhausted(
            f"Gemini quota exhausted after {_MAX_RETRIES + 1} attempts."
        ) from last_exc

    def _extract_sql(self, raw: str) -> str:
        # Xóa thẻ đóng nếu có
        raw = raw.replace("</SQL>", "").strip()

        match = _SQL_FENCE_PATTERN.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _validate_sql(self, sql: str) -> tuple[bool, str, ErrorCode]:
        normalized = sql.lstrip().upper()

        # DML block (checked first — security takes priority)
        dml_match = _DML_PATTERN.search(sql)
        if dml_match:
            return (
                False,
                f"DML operation blocked: {dml_match.group()}",
                ErrorCode.SQL_BLOCKED,
            )

        # Must be SELECT or CTE
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            return False, "LLM returned non-SELECT output.", ErrorCode.CANNOT_ANSWER

        # BQ dry-run — syntax + schema resolution
        try:
            job_cfg = QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bq_client.query(sql, job_config=job_cfg)
            return True, "", ErrorCode.SUCCESS
        except Exception as e:
            return False, f"BQ dry-run: {e}", ErrorCode.CANNOT_ANSWER

    # def _execute_sql(self, sql: str) -> pd.DataFrame:
    #     job_cfg = bigquery.QueryJobConfig(job_timeout_ms=20_000)
    #     query_job = self.bq_client.query(sql, job_config=job_cfg)
        
    #     # buộc client từ bỏ và ném lỗi TimeoutError sau đúng 20 giây chờ
    #     query_job.result(timeout=20.0) 
        
    #     return query_job.to_dataframe()

    # chatbot_engine.py

    def _execute_sql(self, sql: str) -> tuple[pd.DataFrame, Optional[str]]:
        """
        Returns (dataframe, error_code_str).
        Merges validation + execution into single BQ round-trip.
        """
        # DML check vẫn giữ ở đây (security-first)
        dml_match = _DML_PATTERN.search(sql)
        if dml_match:
            return pd.DataFrame(), ErrorCode.SQL_BLOCKED.value

        normalized = sql.lstrip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            return pd.DataFrame(), ErrorCode.CANNOT_ANSWER.value

        try:
            job_cfg = bigquery.QueryJobConfig(job_timeout_ms=20_000)
            query_job = self.bq_client.query(sql, job_config=job_cfg)
            query_job.result(timeout=20.0)
            return query_job.to_dataframe(), None

        except DeadlineExceeded:
            return pd.DataFrame(), ErrorCode.TIMEOUT.value

        except Exception as e:
            err_str = str(e).lower()
            # BQ schema/syntax error → cannot_answer (không expose detail cho user)
            if any(k in err_str for k in ["unrecognized name", "not found", "syntax error", "invalid"]):
                logger.warning("[bq-execute] Schema/syntax error: %s", e)
                return pd.DataFrame(), ErrorCode.CANNOT_ANSWER.value
            raise  # unexpected → bubble lên chat() handler


    def _build_structured_fallback(self, df: pd.DataFrame) -> str:
        """
        Deterministic markdown table fallback when NL generation fails.
        Used only when Gemini call fails or returns empty — not the primary path.
        """
        if df.empty:
            return _USER_MESSAGES[ErrorCode.NO_DATA]

        display_max = 10
        display_df = df.head(display_max).copy()

        priority_cols = [
            "hospital_name", "state", "status", "occupancy_pct",
            "available_beds", "predicted_pct", "forecast_date", "alert_flag",
        ]
        ordered_cols = [c for c in priority_cols if c in display_df.columns]
        remaining    = [c for c in display_df.columns if c not in ordered_cols]
        display_df   = display_df[ordered_cols + remaining]

        headers   = [str(c).replace("_", " ").title() for c in display_df.columns]
        separator = "|" + "|".join(["---"] * len(headers)) + "|"
        rows = [
            "| " + " | ".join(str(display_df.iloc[i][c]) for c in display_df.columns) + " |"
            for i in range(len(display_df))
        ]

        lines = [
            f"### 📊 Summary\n",
            f"Found **{len(df)}** result(s).\n",
            "### 🏥 Data\n",
            "| " + " | ".join(headers) + " |",
            separator,
            *rows,
        ]

        if len(df) > display_max:
            lines.append(f"\n*…and {len(df) - display_max} more results.*")

        lines.append(
            "\n### 💡 Recommendation\n"
            "Review the results above and apply appropriate operational response."
        )

        return "\n".join(lines)


    def _format_response(
        self,
        user_query: str,
        sql: str,
        df: pd.DataFrame,
    ) -> tuple[str, ErrorCode]:
        if df.empty:
            return _USER_MESSAGES[ErrorCode.NO_DATA], ErrorCode.SUCCESS

        anomaly_warning = self._detect_anomalies(df)

        llm_display_rows = min(5, _MAX_DISPLAY_ROWS)
        result_str = df.head(llm_display_rows).to_csv(index=False)
        if len(df) > llm_display_rows:
            result_str += f"\n[Showing {llm_display_rows} of {len(df)} total rows]"

        sys_instruct = """\
    You are a hospital operations analytics assistant for healthcare managers.

    Respond using EXACTLY this structure (copy the ### headers verbatim):

    ### 📊 Summary
    [1-2 sentences. Lead with the critical finding. **Bold** key numbers and hospital names.]

    ### 🏥 Data
    [Markdown table, max 10 rows, only the most relevant columns.
    If rows are truncated, add: *…and N more hospitals.*]

    ### 💡 Recommendation
    [1-2 sentences. Concrete operational action managers can take now.]

    HARD RULES — never violate:
    - Use ### headers exactly as shown. Never replace them with numbered labels.
    - Occupancy as percentage: 87.3%, never 0.873.
    - Never mention SQL, table names, column names, or database internals.
    - If the data contains an anomaly warning, lead the Summary with it.
    - Total response under 300 words.\
    """

        self._nl_config.system_instruction = sys_instruct
        self._nl_config.temperature = 0.2   # Lower = more consistent formatting

        prompt = (
            f"{anomaly_warning}"           # injected only when anomaly detected
            f"Question: {user_query}\n\n"
            f"Data ({len(df)} total rows, showing first {llm_display_rows}):\n"
            f"```csv\n{result_str}\n```"
        )

        try:
            logger.info(
                "NL formatter invoked | rows=%d | model=%s | anomaly=%s",
                len(df), self.gemini_model, bool(anomaly_warning),
            )

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=self._nl_config,
            )

            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason

            if finish_reason == FinishReason.MAX_TOKENS:
                text = (response.text or "").strip()
                logger.warning("NL formatter MAX_TOKENS | partial_chars=%d", len(text))
                return (text or self._build_structured_fallback(df)), ErrorCode.SUCCESS

            if finish_reason not in (FinishReason.STOP, None):
                logger.warning("NL formatter non-retryable finish reason: %s", finish_reason)
                return self._build_structured_fallback(df), ErrorCode.SUCCESS

            text = (response.text or "").strip()
            if not text:
                logger.warning("NL formatter empty text despite STOP finish.")
                return self._build_structured_fallback(df), ErrorCode.SUCCESS

            return text, ErrorCode.SUCCESS

        except _GEMINI_RETRY_EXCEPTIONS as e:
            logger.warning("[fail-fast] NL formatter quota/network error (%s).", type(e).__name__)
            return self._build_structured_fallback(df), ErrorCode.SUCCESS

        except Exception as e:
            logger.error("NL formatter unexpected error: %s", e, exc_info=True)
            return self._build_structured_fallback(df), ErrorCode.SUCCESS

    def _detect_anomalies(self, df: pd.DataFrame) -> str:
        """
        Deterministic pre-check before NL formatting.
        Returns a warning string to inject into the prompt, or "" if clean.
        Catches obvious data pipeline bugs (e.g. predicted_occupancy 9900%)
        so the LLM doesn't need to guess.
        """
        for col in _PCT_COLUMNS:
            if col not in df.columns:
                continue
            max_val = pd.to_numeric(df[col], errors="coerce").max()
            if pd.notna(max_val) and max_val > _ANOMALY_PCT_THRESHOLD:
                logger.warning(
                    "[anomaly-check] Column '%s' max=%.1f exceeds threshold %.1f — "
                    "likely data pipeline issue.",
                    col, max_val, _ANOMALY_PCT_THRESHOLD,
                )
                return (
                    f"⚠️ DATA ANOMALY DETECTED: Column '{col}' contains value {max_val:.1f}% "
                    f"which exceeds physically possible range. "
                    f"This is a data pipeline issue — NOT a real clinical finding. "
                    f"You MUST flag this prominently in your response instead of treating it as valid data.\n\n"
                )
        return ""

    def _audit_log(
        self,
        conversation_id: str,
        query: str,
        sql: str,
        row_count: int,
        response: str,
        status: str,
        error_message: Optional[str],
        latency_ms: float,
    ) -> None:
        try:
            row = {
                "conversation_id": conversation_id,
                "query":           query,
                "generated_sql":   (sql or "")[:4096],
                "row_count":       row_count,
                "response":        (response or "")[:2048],
                "status":          status,
                "error_message":   (error_message or "")[:1024],
                "latency_ms":      round(latency_ms, 1),
                "created_at":      datetime.now(timezone.utc).isoformat(),
            }
            errors = self.bq_client.insert_rows_json(self.audit_table, [row])
            if errors:
                logger.warning("[audit] Streaming insert errors: %s", errors)
        except Exception as e:
            logger.warning("[audit] Non-critical failure — chat unaffected: %s", e)

    def chat(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
    ) -> dict:
        conversation_id  = conversation_id or str(uuid.uuid4())
        t0               = time.monotonic()
        sql              = ""
        row_count        = 0
        error_code       = ErrorCode.ERROR
        error_detail: Optional[str] = None
        response         = ""

        try:
            logger.info("[%s] Query: %.120s", conversation_id, user_query)

            # ── Gate 0: Scope pre-filter ──────────────────────────────────
            if not self._is_in_scope(user_query):
                error_code = ErrorCode.CANNOT_ANSWER
                response   = _USER_MESSAGES[ErrorCode.CANNOT_ANSWER]
                logger.info("[%s] Scope pre-filter: rejected.", conversation_id)

            else:
                # ── Pass 1: SQL generation with retry ─────────────────────
                raw_sql = self._generate_sql_with_retry(user_query)
                sql     = self._extract_sql(raw_sql)
                logger.info("[%s] SQL (%d chars):\n%s", conversation_id, len(sql), sql)
                logger.info(
                    "[%s] SQL generated | chars=%d | tokens≈%d",
                    conversation_id, len(sql), len(sql) // 4,
                )

                # ── SQL guard ─────────────────────────────────────────────
                valid, guard_error, guard_code = self._validate_sql(sql)
                if not valid:
                    error_code   = guard_code
                    error_detail = guard_error
                    response     = _USER_MESSAGES[error_code]
                    if error_code == ErrorCode.SQL_BLOCKED:
                        sql = ""
                    logger.warning(
                        "[%s] SQL guard blocked [%s]: %s",
                        conversation_id, error_code, guard_error,
                    )
                else:
                    # ── BQ Execution ──────────────────────────────────────
                    df, exec_error = self._execute_sql(sql)
                    if exec_error:
                        error_code = ErrorCode(exec_error)
                        response = _USER_MESSAGES[error_code]
                        if error_code == ErrorCode.SQL_BLOCKED:
                            sql = ""
                    else:
                        # ── Pass 2: NL formatting ─────────────────────────
                        # response   = self._format_response(user_query, sql, df)
                        # error_code = ErrorCode.SUCCESS
                        response, error_code = self._format_response(user_query, sql, df)

        except DeadlineExceeded as e:
            error_code   = ErrorCode.TIMEOUT
            error_detail = str(e)
            response     = _USER_MESSAGES[ErrorCode.TIMEOUT]
            logger.warning("[%s] BQ timeout: %s", conversation_id, e)

        except ResourceExhausted as e:
            error_code   = ErrorCode.RATE_LIMITED
            error_detail = str(e)
            response     = _USER_MESSAGES[ErrorCode.RATE_LIMITED]
            logger.error("[%s] Gemini rate limit exhausted: %s", conversation_id, e)

        except Exception as e:
            error_code   = ErrorCode.ERROR
            error_detail = str(e)
            response     = _USER_MESSAGES[ErrorCode.ERROR]
            logger.error("[%s] Unexpected error: %s", conversation_id, e, exc_info=True)

        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[%s] Done | status=%s | rows=%d | latency=%.0fms",
            conversation_id, error_code.value, row_count, latency_ms,
        )

        self._audit_log(
            conversation_id = conversation_id,
            query           = user_query,
            sql             = sql,
            row_count       = row_count,
            response        = response,
            status          = error_code.value,
            error_message   = error_detail,
            latency_ms      = latency_ms,
        )

        return {
            "conversation_id": conversation_id,
            "response":        response,
            "sql":             sql,
            "row_count":       row_count,
            "status":          error_code.value,
        }