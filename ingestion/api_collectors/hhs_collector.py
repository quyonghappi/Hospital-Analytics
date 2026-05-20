# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import logging
from datetime import datetime
from sodapy import Socrata
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = os.getenv("GCP_PROJECT", "project-8e2366a6-d3cc-40ee-9de")
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev")
HHS_DOMAIN = "healthdata.gov"
DATASET_ID = "anag-cw7u"

def fetch_hhs_data():
    logging.info(f"Fetching HHS dataset {DATASET_ID} from {HHS_DOMAIN}")
    client = Socrata(HHS_DOMAIN, None) # Cần truyền App Token nếu chạy production tần suất cao
    
    results = client.get(DATASET_ID, limit=50000)
    df = pd.DataFrame.from_records(results)
    
    logging.info(f"Successfully fetched {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def upload_to_gcs(file_path, destination_blob_name):
    """Upload file local lên GCS"""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BRONZE_BUCKET)
    blob = bucket.blob(destination_blob_name)
    
    logging.info(f"Uploading to gs://{BRONZE_BUCKET}/{destination_blob_name}")
    blob.upload_from_filename(file_path)
    logging.info("Upload complete.")

def main():
    """Luồng chạy chính"""
    logging.info(f"--- Starting HHS Capacity Ingestion for date ---")
    
    df_hhs = fetch_hhs_data()
    
    os.makedirs("/tmp/data", exist_ok=True)
    local_parquet_path = f"/tmp/data/hhs_capacity.parquet"
    
    # Ép kiểu chuỗi để đảm bảo lưu Parquet an toàn, mọi định dạng float/int sẽ do Spark lo
    df_hhs = df_hhs.astype(str)
    df_hhs.to_parquet(local_parquet_path, engine="pyarrow", index=False)
    
    gcs_path = f"hhs/hospital_capacity/hhs_capacity.parquet"
    upload_to_gcs(local_parquet_path, gcs_path)
    
    if os.path.exists(local_parquet_path):
        os.remove(local_parquet_path)
        
    logging.info("--- HHS Capacity Ingestion Finished ---")

if __name__ == "__main__":
    main()