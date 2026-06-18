import anyio  # REQUIRED: was missing — caused NameError at startup
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from engine.chatbot_engine import ChatbotEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("api-main")

load_dotenv()

PROJECT_ID  = os.getenv("PROJECT_ID",  "project-8e2366a6-d3cc-40ee-9de")
REGION      = os.getenv("REGION",      "global")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "config/schema_context.yaml")
AUDIT_TABLE = f"{PROJECT_ID}.ml_observability_dev.chatbot_audit_log"

# Singleton engine — set during lifespan startup, read in endpoint
engine: Optional[ChatbotEngine] = None


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about hospital capacity",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Client-provided UUID for tracking multi-turn context (logging only)",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    sql: str
    row_count: int
    status: str = Field(
        ...,
        description=(
            "'success'       — data returned and NL-formatted\n"
            "'no_data'       — query ran, zero rows matched\n"
            "'cannot_answer' — question outside analytics scope\n"
            "'sql_blocked'   — security violation; sql field will be empty\n"
            "'timeout'       — BQ query exceeded 20s; try a narrower question\n"
            "'rate_limited'  — upstream quota exhausted; retry after a moment\n"
            "'error'         — unexpected internal error"
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise engine and validate dependencies. Shutdown: no-op."""

    # Raise AnyIO thread pool limit for concurrent BQ + Vertex AI I/O
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 60
    logger.info("AnyIO thread pool set to %d tokens", limiter.total_tokens)

    global engine
    try:
        engine = ChatbotEngine(
            project_id=PROJECT_ID,
            region=REGION,
            schema_path=SCHEMA_PATH,
            audit_table=AUDIT_TABLE,
            gemini_model="gemini-3.5-flash",
        )
        logger.info("ChatbotEngine initialised for project=%s", PROJECT_ID)

        # Validate audit table exists — fail fast so ops catches missing DDL
        try:
            engine.bq_client.get_table(AUDIT_TABLE)
            logger.info("Audit table %s validated.", AUDIT_TABLE)
        except Exception:
            logger.error(
                "Audit table %s missing. Run DDL via Airflow/Terraform first.",
                AUDIT_TABLE,
            )
            raise RuntimeError("Missing dependency: Audit Table")

    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        raise

    yield  # App is running

    # Shutdown — BQ and Vertex AI clients have no explicit close()
    logger.info("ChatbotEngine shutting down.")


app = FastAPI(
    title="Hospital Analytics Chatbot API",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Text-to-SQL endpoint: natural language → SQL → BQ → natural language.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Chatbot engine not initialised.")

    try:
        result = engine.chat(
            user_query=request.query,
            conversation_id=request.conversation_id,
        )
        return result
    except Exception as e:
        logger.error("Fatal error in chat endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
async def health_check():
    """
    Liveness + dependency probe.

    Tests BQ and Vertex AI connections inside the live engine instance.
    """
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "offline", "reason": "Engine not loaded"},
        )

    status = {
        "service":    "chatbot-engine",
        "bigquery":   "unknown",
        "vertex_ai":  "unknown",
    }

    # BigQuery probe
    try:
        list(engine.bq_client.query("SELECT 1").result())
        status["bigquery"] = "healthy"
    except Exception as e:
        logger.error("BigQuery health check failed: %s", e)
        status["bigquery"] = f"unhealthy: {e}"

    # Vertex AI / Gemini probe
    try:
        # Sử dụng SDK mới thông qua engine.client
        res = engine.client.models.generate_content(
            model=engine.gemini_model,
            contents="Return only SQL: SELECT 1",
            config=engine._sql_config
        )
        if res.text:
            status["vertex_ai"] = "healthy"
    except Exception as e:
        logger.error("Vertex AI health check failed: %s", e)
        status["vertex_ai"] = f"unhealthy: {e}"

    http_status = (
        200
        if status["bigquery"] == "healthy" and status["vertex_ai"] == "healthy"
        else 503
    )
    return JSONResponse(status_code=http_status, content=status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
    )