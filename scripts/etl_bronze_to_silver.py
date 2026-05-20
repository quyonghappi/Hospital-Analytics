# -*- coding: utf-8 -*-
# etl_bronze_to_silver.py
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, upper, lit, current_timestamp, when, array, to_date,
    coalesce, round, size, filter, to_json, struct, date_format
)
from pyspark.sql.types import IntegerType, FloatType, DateType, StringType

# ==================== CẤU HÌNH BUCKET & PROJECT ====================
BRONZE_BUCKET = "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev"
SILVER_BUCKET = "project-8e2366a6-d3cc-40ee-9de-silver-curated-dev"
QUARANTINE_BUCKET = "project-8e2366a6-d3cc-40ee-9de-quarantine-dev"
PROJECT_ID = "project-8e2366a6-d3cc-40ee-9de"
BQ_DATASET = "hospital_dwh"
BQ_TABLE = "pipeline_quality_log"

# ==================== 1. ĐỌC DỮ LIỆU BRONZE ====================
def read_cms_pos(spark, process_date):
    """Đọc tất cả file CMS POS dạng cms_pos_q*_*.parquet từ Bronze"""
    pattern = f"gs://{BRONZE_BUCKET}/cms/pos/cms_pos_*.parquet"
    df = spark.read.parquet(pattern)
    return df.withColumn("_source_file", lit("cms_pos")) \
             .withColumn("_ingested_at", current_timestamp())

def read_hhs(spark, process_date):
    """Đọc file HHS capacity"""
    path = f"gs://{BRONZE_BUCKET}/hhs/hospital_capacity/hhs_capacity.parquet"
    df = spark.read.parquet(path)
    return df.withColumn("_source_file", lit("hhs")) \
             .withColumn("_ingested_at", current_timestamp())

# ==================== 2. TRANSFORM CMS POS  ====================
def transform_cms_pos(df):
    """
    CMS POS Bronze -> Silver
    Mapping cột:
    PRVDR_NUM -> provider_id
    FAC_NM -> facility_name
    CITY -> city
    STATE_CD -> state
    ZIP_CD -> zip_code
    FIPS_CNTY_CD -> fips_county_code
    CRTFD_BED_CNT -> certified_bed_count (Integer)
    FAC_TYPE_CD -> facility_type_code
    """
    # Đổi tên cột
    rename_map = {
        "PRVDR_NUM": "provider_id",
        "FAC_NM": "facility_name",
        "CITY": "city",
        "STATE_CD": "state",
        "ZIP_CD": "zip_code",
        "FIPS_CNTY_CD": "fips_county_code",
        "CRTFD_BED_CNT": "certified_bed_count",
        "FAC_TYPE_CD": "facility_type_code"
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # Kiểu dữ liệu
    if "certified_bed_count" in df.columns:
        df = df.withColumn("certified_bed_count", col("certified_bed_count").cast(IntegerType()))
    if "zip_code" in df.columns:
        df = df.withColumn("zip_code", col("zip_code").cast(StringType()))
    if "fips_county_code" in df.columns:
        df = df.withColumn("fips_county_code", col("fips_county_code").cast(StringType()))

    # Chuẩn hóa text: trim + upper
    for c in ["facility_name", "city", "state"]:
        if c in df.columns:
            df = df.withColumn(c, trim(upper(col(c))))

    # Xóa duplicate theo provider_id (giữ bản ghi đầu tiên)
    df = df.dropDuplicates(["provider_id"])

    # Metadata
    df = df.withColumn("_processed_at", current_timestamp())

    # Chỉ giữ các cột silver yêu cầu
    keep_cols = [
        "provider_id", "facility_name", "city", "state", "zip_code",
        "fips_county_code", "certified_bed_count", "facility_type_code",
        "_source_file", "_ingested_at", "_processed_at"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*existing_cols)

# ==================== 2b. TRANSFORM HHS (cập nhật theo schema thực tế) ====================
def transform_hhs(df):
    """
    HHS Bronze -> Silver dựa trên schema thực tế.
    Mapping cột:
    - hospital_pk                                   -> provider_id
    - hospital_name                                 -> facility_name
    - state                                         -> state
    - fips_code                                     -> fips_county_code
    - collection_week                               -> report_date
    - total_beds_7_day_avg                          -> inpatient_beds_total
    - inpatient_beds_used_7_day_avg                 -> inpatient_beds_used
    - total_staffed_adult_icu_beds_7_day_avg        -> icu_beds_staffed
    - icu_beds_used_7_day_avg                       -> icu_beds_used (thêm để dùng rule R2)
    - total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg -> covid_patients
    """
    rename_map = {
        "hospital_pk": "provider_id",
        "hospital_name": "facility_name",
        "state": "state",
        "fips_code": "fips_county_code",
        "collection_week": "report_date",
        "total_beds_7_day_avg": "inpatient_beds_total",
        "inpatient_beds_used_7_day_avg": "inpatient_beds_used",
        "total_staffed_adult_icu_beds_7_day_avg": "icu_beds_staffed",
        "icu_beds_used_7_day_avg": "icu_beds_used",
        "total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg": "covid_patients"
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # Tạo cột total_patients (dùng inpatient_beds_used để thay thế cho tổng bệnh nhân)
    if "inpatient_beds_used" in df.columns:
        df = df.withColumn("total_patients", col("inpatient_beds_used"))

    # Ép kiểu số
    numeric_cols = ["inpatient_beds_total", "inpatient_beds_used",
                    "icu_beds_staffed", "icu_beds_used", "covid_patients", "total_patients"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(FloatType()))

    # Chuyển collection_week -> DateType (định dạng "YYYY-MM-DD")
    if "report_date" in df.columns:
        df = df.withColumn("report_date", to_date(col("report_date")))

    # Chuẩn hóa text
    for c in ["facility_name", "state"]:
        if c in df.columns:
            df = df.withColumn(c, trim(upper(col(c))))

    # Tính occupancy_rate = (inpatient_beds_used / inpatient_beds_total) * 100
    df = df.withColumn(
        "occupancy_rate",
        when(col("inpatient_beds_total") > 0,
             round((col("inpatient_beds_used") / col("inpatient_beds_total")) * 100, 2)
        ).otherwise(lit(None))
    )

    # Xóa duplicate theo (provider_id, report_date)
    df = df.dropDuplicates(["provider_id", "report_date"])

    # Metadata
    df = df.withColumn("_processed_at", current_timestamp())

    # Giữ các cột silver cần thiết
    keep_cols = [
        "provider_id", "facility_name", "state", "fips_county_code",
        "report_date", "inpatient_beds_total", "inpatient_beds_used",
        "icu_beds_staffed", "icu_beds_used", "occupancy_rate",
        "covid_patients", "total_patients",
        "_source_file", "_ingested_at", "_processed_at"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*existing_cols)

# ==================== 3. BUSINESS RULES (Task 5) ====================
def apply_business_rules(df, dataset_type):
    """
    Thêm cột _violations (array<string>) và _has_violation (boolean)
    dataset_type: 'hhs' hoặc 'cms'
    """
    violations = []

    # R6: report_date trong tương lai (chung cho cả hai nếu có cột report_date)
    if "report_date" in df.columns:
        violations.append(
            when(col("report_date") > current_timestamp(), lit("R6_future_report_date"))
        )

    # Các rule đặc thù cho HHS
    if dataset_type == "hhs":
        # R1: occupancy_rate > 120
        if "occupancy_rate" in df.columns:
            violations.append(
                when(col("occupancy_rate") > 120, lit("R1_occupancy_over_120"))
            )
        # R2: icu_beds_used > icu_beds_staffed
        if "icu_beds_used" in df.columns and "icu_beds_staffed" in df.columns:
            violations.append(
                when(col("icu_beds_used") > col("icu_beds_staffed"), lit("R2_icu_used_gt_staffed"))
            )
        # R3: icu_beds_staffed > inpatient_beds_total
        if "icu_beds_staffed" in df.columns and "inpatient_beds_total" in df.columns:
            violations.append(
                when(col("icu_beds_staffed") > col("inpatient_beds_total"), lit("R3_icu_gt_total_beds"))
            )
        # R4: inpatient_beds_total = 0 nhưng inpatient_beds_used > 0
        if "inpatient_beds_total" in df.columns and "inpatient_beds_used" in df.columns:
            violations.append(
                when((col("inpatient_beds_total") == 0) & (col("inpatient_beds_used") > 0),
                     lit("R4_zero_beds_but_patients"))
            )
        # R5: covid_patients > total_patients
        if "covid_patients" in df.columns and "total_patients" in df.columns:
            violations.append(
                when(col("covid_patients") > col("total_patients"), lit("R5_covid_gt_total"))
            )

    # Gộp các violation thành array
    if violations:
        # Kết hợp các khiếu nại bằng coalesce (ưu tiên violation đầu tiên)
        violation_expr = coalesce(*violations, lit(None))
        df = df.withColumn(
            "_violations",
            when(violation_expr.isNotNull(), array(violation_expr)).otherwise(array().cast("array<string>"))
        )
    else:
        df = df.withColumn("_violations", array().cast("array<string>"))

    df = df.withColumn("_has_violation", when(size(col("_violations")) > 0, True).otherwise(False))
    return df

# ==================== 4. ROUTE SILVER / QUARANTINE (Task 6) ====================
def write_silver_and_quarantine(df, source_name, process_date):
    """
    Ghi clean records (không vi phạm) vào Silver bucket,
    ghi bad records (có vi phạm) vào Quarantine bucket.
    Trả về số lượng clean và bad.
    """
    clean_df = df.filter(~col("_has_violation"))
    bad_df = df.filter(col("_has_violation"))

    silver_path = f"gs://{SILVER_BUCKET}/silver/{source_name}/{process_date}"
    quarantine_path = f"gs://{QUARANTINE_BUCKET}/{source_name}/{process_date}"

    clean_df.write.mode("overwrite").parquet(silver_path)

    if bad_df.count() > 0:
        bad_df = bad_df.withColumn("_quarantine_source", lit(source_name)) \
                       .withColumn("_quarantine_date", lit(process_date))
        bad_df.write.mode("overwrite").parquet(quarantine_path)
    else:
        print(f"No bad records for {source_name} on {process_date}")

    return clean_df.count(), bad_df.count()

# ==================== 5. GHI LOG CHẤT LƯỢNG VÀO BIGQUERY ====================
def log_to_bigquery(spark, source_name, process_date, total_rows, clean_rows, bad_rows, quarantine_rate):
    """Append một dòng log vào bảng BigQuery"""
    from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, FloatType, TimestampType

    schema = StructType([
        StructField("source", StringType(), True),
        StructField("process_date", DateType(), True),
        StructField("total_rows", IntegerType(), True),
        StructField("clean_rows", IntegerType(), True),
        StructField("quarantine_rows", IntegerType(), True),
        StructField("quarantine_rate", FloatType(), True),
        StructField("logged_at", TimestampType(), True)
    ])

    log_data = [(source_name, datetime.strptime(process_date, "%Y-%m-%d").date(),
                 total_rows, clean_rows, bad_rows, quarantine_rate, datetime.now())]
    log_df = spark.createDataFrame(log_data, schema=schema)

    log_df.write \
        .mode("append") \
        .format("bigquery") \
        .option("table", f"{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}") \
        .option("writeMethod", "direct") \
        .save()

# ==================== 6. MAIN PIPELINE ====================
def run_pipeline(spark, process_date):
    # Đọc dữ liệu
    df_cms = read_cms_pos(spark, process_date)
    df_hhs = read_hhs(spark, process_date)

    # Transform
    df_cms_trans = transform_cms_pos(df_cms)
    df_hhs_trans = transform_hhs(df_hhs)

    # Business rules
    df_cms_valid = apply_business_rules(df_cms_trans, "cms")
    df_hhs_valid = apply_business_rules(df_hhs_trans, "hhs")

    # Route và ghi
    clean_cms, bad_cms = write_silver_and_quarantine(df_cms_valid, "cms_pos", process_date)
    clean_hhs, bad_hhs = write_silver_and_quarantine(df_hhs_valid, "hhs", process_date)

    # Log quality
    total_cms = clean_cms + bad_cms
    total_hhs = clean_hhs + bad_hhs
    if total_cms > 0:
        log_to_bigquery(spark, "cms_pos", process_date, total_cms, clean_cms, bad_cms, bad_cms/total_cms)
    if total_hhs > 0:
        log_to_bigquery(spark, "hhs", process_date, total_hhs, clean_hhs, bad_hhs, bad_hhs/total_hhs)

    print(f"===== Pipeline finished for {process_date} =====")
    print(f"CMS POS: clean={clean_cms}, bad={bad_cms}")
    print(f"HHS: clean={clean_hhs}, bad={bad_hhs}")

if __name__ == "__main__":
    # Lấy tham số ngày từ dòng lệnh (VD: 2026-05-20)
    if len(sys.argv) > 1:
        process_date = sys.argv[1]
    else:
        process_date = datetime.now().strftime("%Y-%m-%d")

    # Khởi tạo Spark Session với GCS connector
    spark = SparkSession.builder \
        .appName("BronzeToSilver_HospitalAnalytics") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .getOrCreate()

    run_pipeline(spark, process_date)
    spark.stop()
