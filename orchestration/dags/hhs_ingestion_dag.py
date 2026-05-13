from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from datetime import timedelta

from ingestion.api_collectors.hhs_collector import main as hhs_upload_main

# Config múi giờ chuẩn VN theo NFR
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

default_args = {
    'owner': 'data_engineer_team',
    'depends_on_past': False, # Không block nếu run trước đó fail
    'email_on_failure': True,
    'email': ['admin@hospital-platform.com'],
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ingest_hhs_capacity_to_bronze',
    default_args=default_args,
    description='Fetch weekly hospital capacity from HHS and load to GCS Bronze',
    schedule_interval='0 2 * * 1',  # Chạy vào 2h sáng t2 hàng tuần
    start_date=pendulum.datetime(2024, 5, 1, tz=local_tz),
    catchup=False, # tránh Airflow tự động trigger run cũ khi vừa bật DAG
    tags=['ingestion', 'hhs', 'bronze'],
) as dag:

    fetch_and_upload_hhs = PythonOperator(
        task_id='fetch_and_upload_hhs',
        python_callable=hhs_upload_main,
        op_kwargs={'execution_date': '{{ ds }}'} 
    )

    fetch_and_upload_hhs