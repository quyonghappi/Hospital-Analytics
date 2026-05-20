# -*- coding: utf-8 -*-
import os
import sys
import requests
import pandas as pd
import logging
from datetime import datetime
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = os.getenv("GCP_PROJECT", "project-8e2366a6-d3cc-40ee-9de")
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev")

CMS_POS_URL = "https://data.cms.gov/data-api/v1/dataset/8f6da2b1-f719-40c2-8f73-2b7ecb7fb42a/data"

def fetch_cms_data():
    logging.info(f"Fetching CMS POS data from {CMS_POS_URL}")
    response = requests.get(CMS_POS_URL, params={"limit": 500000})
    response.raise_for_status() # Bắn lỗi ngay nếu HTTP request fail
    
    data = response.json()
    df = pd.DataFrame(data)
    logging.info(f"Successfully fetched {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def extract_quarter_year_from_data(df):
    logging.info("Extracting Quarter and Year directly from data payload...")
    
    # Lấy chuỗi ngày mẫu từ dòng đầu tiên có dữ liệu hợp lệ để định danh cho toàn bộ snapshot
    valid_dates = df['CRTFCTN_DT'].dropna()
    if valid_dates.empty:
        logging.error("Không tìm thấy trường 'CRTFCTN_DT' hoặc trường này bị rỗng trong dữ liệu.")
        sys.exit(1)
        
    sample_date_str = str(valid_dates.iloc[0]).strip()
    
    try:
        dt = datetime.strptime(sample_date_str[:8], "%Y%m%d")
        quarter = (dt.month - 1) // 3 + 1
        return f"q{quarter}_{dt.year}"
    except Exception as e:
        # Fallback an toàn nếu chuỗi không đúng định dạng YYYYMMDD
        logging.warning(f"Không thể parse ngày từ '{sample_date_str}' do lỗi: {e}. Đang dùng chế độ Fallback.")
        from datetime import datetime
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return f"q{quarter}_{now.year}"

def upload_to_gcs(file_path, destination_blob_name):
    """Upload file local lên GCS"""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BRONZE_BUCKET)
    blob = bucket.blob(destination_blob_name)
    
    logging.info(f"Uploading to gs://{BRONZE_BUCKET}/{destination_blob_name}")
    blob.upload_from_filename(file_path)
    logging.info("Upload complete.")

def main():
    """Luồng chạy chính - Không phụ thuộc tham số ngày truyền vào hệ thống"""
    df_pos = fetch_cms_data()
    
    from datetime import datetime # import bổ sung cho hàm extract
    quarter_year_str = extract_quarter_year_from_data(df_pos)
    logging.info(f"Data payload identified as: {quarter_year_str.upper()}")
    
    os.makedirs("/tmp/data", exist_ok=True)
    local_parquet_path = f"/tmp/data/cms_pos_{quarter_year_str}.parquet"
    
    df_pos = df_pos.astype(str)
    df_pos.to_parquet(local_parquet_path, engine="pyarrow", index=False)
    
    gcs_path = f"cms/pos/cms_pos_{quarter_year_str}.parquet"
    upload_to_gcs(local_parquet_path, gcs_path)
    
    if os.path.exists(local_parquet_path):
        os.remove(local_parquet_path)
        
    logging.info(f"--- CMS POS Ingestion Finished for {quarter_year_str.upper()} ---")

if __name__ == "__main__":
    main()