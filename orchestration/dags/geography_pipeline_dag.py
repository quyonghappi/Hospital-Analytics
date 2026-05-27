from datetime import datetime, timedelta
from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator
)

from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

geography_silver_dataset = Dataset("gcs://lake/silver/geography")

PROJECT_ID = Variable.get("gcp_project", "project-8e2366a6-d3cc-40ee-9de")
REGION = Variable.get("gcp_region", "asia-southeast1")
BRONZE_BUCKET = Variable.get("bronze_bucket", "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev")
SILVER_BUCKET = Variable.get("silver_bucket", "project-8e2366a6-d3cc-40ee-9de-silver-curated-dev")
QUARANTINE_BUCKET = Variable.get("quarantine_bucket", "project-8e2366a6-d3cc-40ee-9de-quarantine")

# 2. FIXED: Đổi tên cluster để cô lập tài nguyên cho domain Geography
CLUSTER_NAME = f"geo-spark-{{{{ ds_nodash }}}}"

CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 50},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 50},
    },
    "software_config": {
        "image_version": "2.1-debian11"
    }
}

default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='geography_bronze_to_silver_pipeline',
    default_args=default_args,
    description='Ingest and process Geography data',
    schedule_interval=None, # Chạy tự động bằng event hoặc thủ công do dữ liệu hiếm khi thay đổi
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['geography', 'silver'],
) as dag:
    
    ingest_geography_bronze = BashOperator(
        task_id='ingest_geography_bronze',
        bash_command=f'python /ingestion/api_collectors/geography_collector.py --exec_date {{{{ ds }}}}',
    )

    create_cluster = DataprocCreateClusterOperator(
        task_id="create_ephemeral_cluster",
        project_id=PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
        cluster_name=CLUSTER_NAME,
    )

    pyspark_geography_job = {
        "reference": {"project_id": PROJECT_ID},
        "placement": {"cluster_name": CLUSTER_NAME},
        "pyspark_job": {
            'main_python_file_uri': f'gs://{PROJECT_ID}-code-bucket/scripts/process_geography_silver.py',
            "args": [
                f"--exec_date={{{{ ds }}}}",
                f"--bronze_bucket={BRONZE_BUCKET}",
                f"--silver_bucket={SILVER_BUCKET}",
                f"--quarantine_bucket={QUARANTINE_BUCKET}",
            ],
        },
    }

    process_geography_silver = DataprocSubmitJobOperator(
        task_id='process_geography_silver',
        job=pyspark_geography_job,
        region=REGION,
        project_id=PROJECT_ID,
        # 3. FIXED: Emit dataset ra ngoài để thông báo cho DAG Dataform và DAG Population
        outlets=[geography_silver_dataset], 
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_ephemeral_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME,
        trigger_rule=TriggerRule.ALL_DONE, 
    )

    ingest_geography_bronze >> create_cluster >> process_geography_silver >> delete_cluster