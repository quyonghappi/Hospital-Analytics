# wait for Geography and Population pipeline to complete before running Dataform transformations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.datasets import Dataset
from airflow.providers.google.cloud.operators.dataform import (
    DataformCreateCompilationResultOperator,
    DataformCreateWorkflowInvocationOperator,
)

population_silver_dataset = Dataset("gcs://lake/silver/population")
geography_silver_dataset = Dataset("gcs://lake/silver/geography")

PROJECT_ID = Variable.get("gcp_project", "project-8e2366a6-d3cc-40ee-9de")
REGION = Variable.get("gcp_region", "asia-southeast1")
DATAFORM_REPO = Variable.get("dataform_repository", "hospital-analytics-transformations") 
DATAFORM_WORKSPACE = Variable.get("dataform_workspace", "dev")
ENVIRONMENT = Variable.get("deployment_environment", "dev")

default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['group04@gmail.com'], # Alert Engine endpoint
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG tự động kích hoạt khi both dataset dân số và địa lý được cập nhật xong ở silver
with DAG(
    dag_id='dwh_dataform_dimension_build',
    default_args=default_args,
    description='Triggers Dataform to load Dimensions into BigQuery Gold Layer',
    schedule=[population_silver_dataset, geography_silver_dataset], 
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['dataform', 'bigquery', 'gold', 'dimension'],
) as dag:

    compile_dataform = DataformCreateCompilationResultOperator(
        task_id="compile_dataform_workspace",
        project_id=PROJECT_ID,
        region=REGION,
        repository_id=DATAFORM_REPO,
        compilation_result={
            "code_compilation_config": {
                "vars": {
                    "data_date": "{{ ds }}", 
                    "env": ENVIRONMENT
                }
            }, 
            "workspace": (
                f"projects/{PROJECT_ID}/locations/{REGION}"
                f"/repositories/{DATAFORM_REPO}/workspaces/dev"
            ), 
        },
    )

    execute_dataform = DataformCreateWorkflowInvocationOperator(
        task_id="execute_dataform_workflow",
        project_id=PROJECT_ID,
        region=REGION,
        repository_id=DATAFORM_REPO,
        asynchronous=False, # BẮT BUỘC: Chuyển sang False để Airflow block và chờ cho đến khi Dataform chạy xong hoàn toàn
        workflow_invocation={
            "compilation_result": "{{ task_instance.xcom_pull('compile_dataform_workspace')['name'] }}",
            "invocation_config": {
                "included_tags": ["dimension"],
                "fully_refresh_incremental_tables_enabled": False
            }
        },
    )

    compile_dataform >> execute_dataform