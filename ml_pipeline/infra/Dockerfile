FROM python:3.11-slim AS base

# System deps: gcc/g++ for xgboost native ext, libgomp1 for lightgbm openmp
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY infra/requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY src/ ./src/

# PYTHONPATH ensures `from xai_shap import ...` resolves from /app/src
# when scripts are invoked as `python src/inference.py ...`
ENV PYTHONPATH=/app/src

# Vertex AI Custom Job pattern:
#   container_spec.args = ["src/inference.py", "--project_id", ...]
# → resolves to: python src/inference.py --project_id ...
ENTRYPOINT ["python"]