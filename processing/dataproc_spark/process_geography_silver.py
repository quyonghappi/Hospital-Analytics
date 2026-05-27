from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, FloatType
import argparse

def process_dim_geography(exec_date: str, bronze_bucket: str, silver_bucket: str, quarantine_bucket: str):
    # Khởi tạo Spark Session tối ưu hóa tính toán trên đám mây
    spark = SparkSession.builder \
        .appName("Silver_DimGeography") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    
    # ── 1. Load Bronze Paths (Đọc từ Bronze Bucket) ──────────────────────────
    acs_path = f"gs://{bronze_bucket}/geography/census_acs/{exec_date}/data.parquet"
    gaz_path = f"gs://{bronze_bucket}/geography/gazetteer/{exec_date}/data.parquet"
    hrr_path = f"gs://{bronze_bucket}/geography/dartmouth_hrr/{exec_date}/data.parquet"
    
    print(f"Reading input matrices from Bronze GCS: {bronze_bucket}")
    df_acs = spark.read.parquet(acs_path)
    df_gaz = spark.read.parquet(gaz_path)
    df_hrr = spark.read.parquet(hrr_path)

    # ── 2. Clean & Join (Left Join on ACS as Ground Truth) ───────────────────
    df_acs = df_acs.withColumn(
        "county_name_clean", 
        F.regexp_replace(F.col("county_name"), "(?i)\\s+(County|Parish|Borough|Census Area|Municipality)$", "")
    )

    df_joined = df_acs.alias("acs") \
        .join(df_gaz.alias("gaz"), "county_fips", "left") \
        .join(df_hrr.alias("hrr"), "county_fips", "left")

    # ── 3. Projection & Schema Enforcement ───────────────────────────────────
    df_silver_base = df_joined.select(
        F.col("county_fips").cast(StringType()),
        F.col("county_name_clean").alias("county_name").cast(StringType()),
        F.col("state_code").cast(StringType()),
        F.col("state_name").cast(StringType()),
        F.col("hrr_region").cast(StringType()),
        F.col("area_km2").cast(FloatType()),
        # Audit columns
        F.current_timestamp().alias("ingestion_timestamp")
    )

    # ── 4. Quality Gate (Quarantine Logic) ───────────────────────────────────
    # Rule: county_fips phải đúng 5 chữ số và state_code không được null
    dq_condition = (
        F.col("county_fips").isNotNull() & 
        F.col("county_fips").rlike("^\\d{5}$") &
        F.col("state_code").isNotNull()
    )

    df_valid = df_silver_base.filter(dq_condition)
    
    df_quarantine = df_silver_base.filter(~dq_condition).withColumn(
        "error_reason", 
        F.lit("DQ_FAIL: Invalid FIPS or missing state_code")
    )

    # ── 5. Write to GCS (Idempotent Overwrite cho từng vùng đích riêng biệt) ──
    silver_out = f"gs://{silver_bucket}/geography/dim_geography/exec_date={exec_date}"
    quarantine_out = f"gs://{quarantine_bucket}/geography/dim_geography/exec_date={exec_date}"

    # .coalesce(1) giúp gộp dữ liệu thành 1 file parquet tập trung thay vì chia nhỏ 
    print(f"Writing clean curated records to Silver GCS: {silver_bucket}")
    df_valid.coalesce(1).write.mode("overwrite").parquet(silver_out)
    
    # Chỉ quét hành động và ghi vào vùng Quarantine nếu xuất hiện bản ghi lỗi
    quarantine_count = df_quarantine.count()
    if quarantine_count > 0:
        print(f"WARNING: Found {quarantine_count} bad records. Quarantining...")
        df_quarantine.coalesce(1).write.mode("overwrite").parquet(quarantine_out)
        print(f"-> Quarantined location: {quarantine_out}")
    else:
        print("Success: 100% data passed Data Quality gate!")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec_date", required=True, help="Format: YYYY-MM-DD")
    parser.add_argument("--bronze_bucket", required=True)
    parser.add_argument("--silver_bucket", required=True)
    parser.add_argument("--quarantine_bucket", required=True)
    args = parser.parse_args()
    
    process_dim_geography(
        exec_date=args.exec_date,
        bronze_bucket=args.bronze_bucket,
        silver_bucket=args.silver_bucket,
        quarantine_bucket=args.quarantine_bucket
    )