from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType
import argparse

# Census ACS age 65+ columns (12 bands: 20-25 male + 44-49 female in B01001)
AGE_MALE_65_PLUS = [f"age_raw_B01001_{i:03d}E" for i in range(20, 26)]
AGE_FEM_65_PLUS  = [f"age_raw_B01001_{i:03d}E" for i in range(44, 50)]
ALL_AGE_65_COLS  = AGE_MALE_65_PLUS + AGE_FEM_65_PLUS

def clean_numeric(col_name, df_cols):
    """Utility format text to float, handle commas and hidden chars"""
    if col_name not in df_cols:
        return F.lit(None).cast(FloatType())
    return F.regexp_replace(F.col(col_name).cast(StringType()), r'[^0-9.\-]', '').cast(FloatType())

def safe_col(col_name, df_cols):
    """Safe column retrieval to avoid AnalysisException for missing years"""
    if col_name not in df_cols:
        return F.lit(None).cast(StringType())
    return F.col(col_name).cast(StringType())


def process_population_silver(exec_date: str, bronze_bucket: str, silver_bucket: str,
                               quarantine_bucket: str, geography_silver_bucket: str):
    spark = (
        SparkSession.builder
        .appName("Silver_DimPopulation")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .getOrCreate()
    )
    
    print("[LOAD] Reading bronze sources...")

    df_acs = spark.read.parquet(
        f"gs://{bronze_bucket}/population/census_acs/{exec_date}/data.parquet"
    )
    df_ahrf = spark.read.parquet(
        f"gs://{bronze_bucket}/population/hrsa_ahrf/{exec_date}/data.parquet"
    )
    df_rucc = spark.read.parquet(
        f"gs://{bronze_bucket}/population/usda_rucc/{exec_date}/data.parquet"
    )
    df_ilinet = spark.read.parquet(
        f"gs://{bronze_bucket}/population/cdc_ilinet/{exec_date}/data.parquet"
    )
    # lấy area_km2 tính population_density
    df_geo_silver = spark.read.parquet(
        f"gs://{geography_silver_bucket}/geography/dim_geography/exec_date={exec_date}"
    ).select("county_fips", "area_km2")

    # Transform ACS: tính derived metrics
    # Tổng dân số 65+
    age_sum_expr = sum(F.col(c) for c in ALL_AGE_65_COLS if c in df_acs.columns)

    df_acs_clean = df_acs \
        .withColumn("pop_65_plus", age_sum_expr) \
        .withColumn(
            "pop_65_plus_pct",
            F.when(
                F.col("total_population_raw") > 0,
                F.round(F.col("pop_65_plus") / F.col("total_population_raw") * 100, 4)
            ).otherwise(F.lit(None).cast(FloatType()))
        ) \
        .withColumn(
            "poverty_rate_pct",
            F.when(
                F.col("poverty_universe_raw") > 0,
                F.round(F.col("pop_below_poverty_raw") / F.col("poverty_universe_raw") * 100, 4)
            ).otherwise(F.lit(None).cast(FloatType()))
        ) \
        .withColumn(
            "uninsured_rate_pct",
            F.when(
                F.col("insurance_universe_raw") > 0,
                F.round(
                    (F.col("uninsured_19_64_raw") + F.col("uninsured_65_plus_raw"))
                    / F.col("insurance_universe_raw") * 100, 4
                )
            ).otherwise(F.lit(None).cast(FloatType()))
        ) \
        .select(
            F.col("county_fips").cast(StringType()),
            F.col("census_year").cast(IntegerType()),
            F.col("total_population_raw").alias("total_population").cast(IntegerType()),
            "pop_65_plus_pct",
            F.col("median_household_income_raw").alias("median_household_income"),
            "poverty_rate_pct",
            "uninsured_rate_pct",
        )

    # Transform AHRF: physicians & nurses per 100k
    ahrf_cols = df_ahrf.columns
    
    df_ahrf_clean = df_ahrf.withColumn(
        # Chuẩn hóa mã địa lý FIPS từ nhiều chuẩn tên khác nhau của AHRF
        "county_fips",
        F.coalesce(
            F.concat(safe_col("f00011", ahrf_cols), safe_col("f00012", ahrf_cols)),
            F.concat(safe_col("fips_st", ahrf_cols), safe_col("fips_cnty", ahrf_cols))
        )
    ).withColumn(
        "physicians_raw",
        F.coalesce(
            clean_numeric("stgh_rn_ft_incl_nh_23", ahrf_cols),              # 2025 (Theo notebook)
            clean_numeric("phys_nf_prim_care_pc_exc_rsdt_22", ahrf_cols),   # 2024
            clean_numeric("stgh_fte_phys_dent_incl_nh_21", ahrf_cols),      # 2023
            clean_numeric("f1130820", ahrf_cols),                           # 2022
            clean_numeric("f1130819", ahrf_cols),                           # 2021
            clean_numeric("f1130818", ahrf_cols)                            # 2020
        )
    ).withColumn(
        "nurses_raw",
        F.coalesce(
            clean_numeric("aprn_npi_24", ahrf_cols),                        # 2025
            clean_numeric("aprn_npi_23", ahrf_cols),                        # 2024
            # 2023: Tổng 2 cột
            clean_numeric("stgh_rn_ft_incl_nh_21", ahrf_cols) + clean_numeric("aprn_npi_22", ahrf_cols), 
            clean_numeric("f1464621", ahrf_cols),                           # 2022
            clean_numeric("f1464620", ahrf_cols),                           # 2021
            clean_numeric("f1464619", ahrf_cols)                            # 2020
        )
    ).withColumn(
        "census_year", F.col("data_year").cast(IntegerType()) # data_year set từ Collector
    ).select(
        "county_fips", "census_year", "physicians_raw", "nurses_raw"
    ).dropDuplicates(["county_fips", "census_year"])

    # Transform RUCC: rural/urban classification
    # broadcast join
    df_rucc_clean = df_rucc.select(
        F.col("county_fips").cast(StringType()),
        F.col("rucc_code_raw").cast(IntegerType()).alias("rucc_code"),
        F.col("rucc_description"),
    ).dropDuplicates(["county_fips"])

    # Transform ILINet: aggregate flu activity level lên state-level ────
    # ILINet không có county FIPS → aggregate per state per season_year
    # Sau đó join với ACS qua state_fips prefix
    df_flu = df_ilinet \
        .groupBy("state_name", "season_start_year") \
        .agg(
            F.avg("activity_level_raw").alias("avg_flu_activity_level"),
            F.max("activity_level_raw").alias("peak_flu_activity_level"),
        ) \
        .withColumnRenamed("season_start_year", "census_year")

    # Join tất cả sources, ACS làm ground truth
    # Broadcast nhỏ để tối ưu cost
    df_joined = df_acs_clean \
        .join(df_ahrf_clean, ["county_fips", "census_year"], "left") \
        .join(F.broadcast(df_rucc_clean), "county_fips", "left") \
        .join(F.broadcast(df_geo_silver), "county_fips", "left")

    # population_density = total_population / area_km2
    df_joined = df_joined.withColumn(
        "population_density",
        F.when(
            F.col("area_km2").isNotNull() & (F.col("area_km2") > 0),
            F.round(
                F.col("total_population").cast(FloatType()) /
                F.col("area_km2").cast(FloatType()),
                4
            )
        ).otherwise(F.lit(None).cast(FloatType()))
    )

    # physicians_per_100k
    df_joined = df_joined.withColumn(
        "physicians_per_100k",
        F.when(
            (F.col("total_population").isNotNull()) &
            (F.col("total_population") > 0) &
            (F.col("physicians_raw").isNotNull()),
            F.round(
                (
                    F.col("physicians_raw").cast(FloatType()) /
                    F.col("total_population").cast(FloatType())
                ) * 100000,
                4
            )
        ).otherwise(F.lit(None).cast(FloatType()))
    )

    # nurses_per_100k
    df_joined = df_joined.withColumn(
        "nurses_per_100k",
        F.when(
            (F.col("total_population").isNotNull()) &
            (F.col("total_population") > 0) &
            (F.col("nurses_raw").isNotNull()),
            F.round(
                (
                    F.col("nurses_raw").cast(FloatType()) /
                    F.col("total_population").cast(FloatType())
                ) * 100000,
                4
            )
        ).otherwise(F.lit(None).cast(FloatType()))
    )

    df_joined = df_joined.drop("area_km2")
    # ILINet join qua state prefix — cần state_name lookup từ geography
    # Defer flu join sang gold/dim nếu không có state_name trong ACS bronze
    # ACS bronze có "state" column (2-digit FIPS prefix)

    df_silver = df_joined.select(
        "county_fips",
        "census_year",
        "total_population",
        "pop_65_plus_pct",
        "population_density",
        "median_household_income",
        "poverty_rate_pct",
        "uninsured_rate_pct",
        "physicians_per_100k",
        "nurses_per_100k",
        "rucc_code",
        "rucc_description",
        F.current_timestamp().alias("ingestion_timestamp"),
        F.lit(exec_date).alias("exec_date"),
    )

    # Quality Gate 
    dq_condition = (
        F.col("county_fips").isNotNull() &
        F.col("county_fips").rlike(r"^\d{5}$") &
        F.col("census_year").isNotNull() &
        F.col("total_population").isNotNull() &
        (F.col("total_population") >= 0)
    )

    df_valid = df_silver.filter(dq_condition)
    df_quarantine = df_silver.filter(~dq_condition).withColumn(
        "error_reason", F.lit("DQ_FAIL: invalid FIPS, null year, or negative population")
    )

    # Write Silver
    silver_out     = f"gs://{silver_bucket}/population/dim_population/exec_date={exec_date}"
    quarantine_out = f"gs://{quarantine_bucket}/population/dim_population/exec_date={exec_date}"

    valid_count = df_valid.count()
    bad_count   = df_quarantine.count()
    total       = valid_count + bad_count

    print(f"[QG] Total: {total} | Valid: {valid_count} | Bad: {bad_count}")

    # Hard stop nếu > 20% records bị quarantine
    if total > 0 and (bad_count / total) > 0.20:
        raise RuntimeError(
            f"[QG] CRITICAL: {bad_count}/{total} ({bad_count/total:.1%}) records failed DQ. "
            "Stopping pipeline to prevent bad data propagation."
        )

    df_valid.coalesce(1).write.mode("overwrite").parquet(silver_out)
    print(f"[WRITE] Silver: {silver_out}")

    if bad_count > 0:
        df_quarantine.coalesce(1).write.mode("overwrite").parquet(quarantine_out)
        print(f"[QUARANTINE] {quarantine_out}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec_date",               required=True)
    parser.add_argument("--bronze_bucket",            required=True)
    parser.add_argument("--silver_bucket",            required=True)
    parser.add_argument("--quarantine_bucket",        required=True)
    parser.add_argument("--geography_silver_bucket",  required=True)
    args = parser.parse_args()

    process_population_silver(
        exec_date=args.exec_date,
        bronze_bucket=args.bronze_bucket,
        silver_bucket=args.silver_bucket,
        quarantine_bucket=args.quarantine_bucket,
        geography_silver_bucket=args.geography_silver_bucket,
    )