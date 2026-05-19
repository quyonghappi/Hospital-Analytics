from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import DataprocCreateBatchOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.dataform import (
    DataformCreateCompilationResultOperator,
    DataformCreateWorkflowInvocationOperator,
)
from datetime import datetime, timedelta
import os
from airflow.models import Variable

# --- CONFIGURATIONS ---
PROJECT_ID = "project-8e2366a6-d3cc-40ee-9de"
REGION = "asia-southeast1"
DATAFORM_REPOSITORY = "hospital_analytics_repo"
ENV = Variable.get("SYSTEM_ENV", default_var="dev")
BRONZE_BUCKET = f"hospital_lake_bronze_{ENV}"

DATA_SOURCES = ['cms', 'hhs', 'census', 'hrsa']

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': True, # NFR-R-01 Target
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30), # Dead-man switch
}

# --- HELPER FUNCTIONS ---
def run_ingestion(source: str, execution_date: str):
    """
    Localized import to prevent Airflow Parse-time timeout & ModuleNotFoundError.
    Assumes scripts are at: orchestration/include/api_collectors/<source>_collector.py
    """
    import importlib
    # Dynamically load the correct collector based on source
    collector = importlib.import_module(f"include.api_collectors.{source}_collector")
    # Trigger the main logic (Must write Parquet to gs://lake/raw/<source>/dt={{ds}}/)
    collector.main(execution_date)

with DAG(
    'hospital_utilization_etl',
    default_args=default_args,
    description='Batch ELT pipeline for Hospital Resource Utilization',
    schedule_interval='0 2 * * *', # Daily at 2 AM
    start_date=datetime(2024, 1, 1), # Static date - NO days_ago()
    catchup=False,
    max_active_runs=1, # Prevent concurrency issues on Dataform
    tags=['core', 'healthcare', 'elt'],
) as dag:

    # 1. INGESTION LAYER (Bronze)
    with TaskGroup("ingestion_bronze") as ingestion_bronze:
        ingest_tasks = []
        for src in DATA_SOURCES:
            task = PythonOperator(
                task_id=f'ingest_{src}_api',
                python_callable=run_ingestion,
                op_kwargs={'source': src, 'execution_date': '{{ ds }}'},
            )
            ingest_tasks.append(task)

    # 2. PROCESSING LAYER (Bronze -> Silver)
    with TaskGroup("processing_silver") as processing_silver:
        # Dataproc Serverless (Batch) Configuration
        pyspark_batch_config = {
            "pyspark_batch": {
                "main_python_file_uri": f"gs://{PROJECT_ID}-artifacts/scripts/pyspark_cleanse.py",
                "args": [
                    "--execution_date={{ ds }}",
                    f"--bronze_path=gs://{BUCKET_NAME}/raw/",
                    f"--silver_path=gs://{BUCKET_NAME}/silver/",
                    f"--quarantine_path=gs://{BUCKET_NAME}/quarantine/",
                    "--max_error_threshold=0.20"
                ],
            },
            "environment_config": {
                "execution_config": {
                    "subnetwork_uri": "default" # Adjust if using Custom VPC
                }
            }
        }
        
        task_pyspark_clean = DataprocCreateBatchOperator(
            task_id='pyspark_quality_and_clean_serverless',
            project_id=PROJECT_ID,
            region=REGION,
            batch=pyspark_batch_config,
            batch_id='hospital-etl-{{ ds_nodash }}',
        )

    # 3. LOAD TO WAREHOUSE (Silver -> BQ Staging)
    with TaskGroup("load_bq_staging") as load_bq_staging:
        load_tasks = []
        for src in DATA_SOURCES:
            task = GCSToBigQueryOperator(
                task_id=f'load_{src}_to_bq',
                bucket=BUCKET_NAME,
                source_objects=[f'silver/{src}/dt={{{{ ds }}}}/*.parquet'],
                destination_project_dataset_table=f'{PROJECT_ID}.staging.{src}_{{{{ ds_nodash }}}}',
                source_format='PARQUET',
                write_disposition='WRITE_TRUNCATE', # Ensure idempotency
                autodetect=True,
            )
            load_tasks.append(task)

    # 4. WAREHOUSE TRANSFORMATION (Dataform Star Schema)
    with TaskGroup("transform_dataform") as transform_dataform:
        compile_dataform = DataformCreateCompilationResultOperator(
            task_id="compile_dataform",
            project_id=PROJECT_ID,
            region=REGION,
            repository_id=DATAFORM_REPOSITORY,
            compilation_result={
                "git_commitish": "main",
                "workspace": "production" # Assuming standard production workspace
            },
        )
        
        invoke_dataform = DataformCreateWorkflowInvocationOperator(
            task_id="invoke_dataform",
            project_id=PROJECT_ID,
            region=REGION,
            repository_id=DATAFORM_REPOSITORY,
            workflow_invocation={
                "compilation_result": "{{ task_instance.xcom_pull('transform_dataform.compile_dataform')['name'] }}"
            },
        )
        
        compile_dataform >> invoke_dataform

    # DAG OVERALL DEPENDENCIES
    ingestion_bronze >> processing_silver >> load_bq_staging >> transform_dataform