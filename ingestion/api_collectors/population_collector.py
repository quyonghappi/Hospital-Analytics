# -*- coding: utf-8 -*-
"""
Sources → Bronze paths:
  Census ACS API   → population/census_acs/{exec_date}/data.parquet
  HRSA AHRF        → population/hrsa_ahrf/{exec_date}/data.parquet
  CDC FluSurv-NET  → population/cdc_fluview/{exec_date}/data.parquet
  USDA RUCC        → population/usda_rucc/{exec_date}/data.parquet
"""
import io
import os
import re
import logging
import requests
import pandas as pd
import pyreadstat
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


PROJECT_ID    = os.getenv("GCP_PROJECT", "project-8e2366a6-d3cc-40ee-9de")
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev")
CENSUS_KEY    = os.getenv("CENSUS_API_KEY", "937589a4e18bf415305ccf41b6e8aa6ee0fcc258")
DATA_DIR      = os.getenv("DATA_DIR", r"C:\Users\HP\Documents\prj-hospital-analytics\downloaded_data")

AGE_MALE_65_PLUS = [f"B01001_{i:03d}E" for i in range(20, 26)]
AGE_FEM_65_PLUS  = [f"B01001_{i:03d}E" for i in range(44, 50)]
AGE_COLS         = AGE_MALE_65_PLUS + AGE_FEM_65_PLUS
ACS_YEARS        = [2020, 2021, 2022, 2023]

AHRF_VINTAGES = [
    {
        "vintage"      : "2020-2021",
        "data_year"    : 2020,
        "fmt"          : "sas7bdat",
        "data_file"    : os.path.join(DATA_DIR, "AHRF_2020-2021_SAS", "AHRF2021.sas7bdat"),
    },
    {
        "vintage"      : "2021-2022",
        "data_year"    : 2021,
        "fmt"          : "sas7bdat",
        "data_file"    : os.path.join(DATA_DIR, "AHRF_2021-2022_SAS", "ahrf2022.sas7bdat"),
    },
    {
        "vintage"      : "2022-2023",
        "data_year"    : 2022,
        "fmt"          : "csv",
        "data_file"    : os.path.join(DATA_DIR, "AHRF_2022-2023_CSV", "ahrf2023.csv"),
    },
    {
        "vintage"      : "2023-2024",
        "data_year"    : 2023,
        "fmt"          : "csv",
        "data_file"    : os.path.join(DATA_DIR, "AHRF 2023-2024 CSV", "ahrf2024_Feb2025.csv"),
    },
    {
        "vintage"      : "2024-2025",
        "data_year"    : 2024,
        "fmt"          : "csv",
        "data_file"    : os.path.join(DATA_DIR, "AHRF_2024-2025_CSV", "AHRF2025.csv"),
    },
]

FLUVIEW_LOCAL  = os.path.join(DATA_DIR, "FluSurv-NET.csv")
RUCC_LOCAL     = os.path.join(DATA_DIR, "rucc2023.csv")
ILINET_LOCAL   = os.path.join(DATA_DIR, "ilinet.csv")

def upload_parquet_to_gcs(df: pd.DataFrame, gcs_path: str) -> None:
    buf = io.BytesIO()
    # Sử dụng nén snappy để giảm tối đa kích thước truyền tải qua mạng
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = buf.tell() / (1024 * 1024)
    buf.seek(0)
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BRONZE_BUCKET)
    blob = bucket.bucket.blob(gcs_path) if hasattr(bucket, 'bucket') else bucket.blob(gcs_path)
    
    # Cấu hình chunk_size
    # giúp tránh nghẽn băng thông mạng local khi đẩy dữ liệu lên Cloud
    if file_size_mb > 10:
        blob.chunk_size = 5 * 1024 * 1024  # 5 MB chunk
        logging.info(f"Large dataset detected ({file_size_mb:.2f} MB). Resumable upload enabled with 5MB chunks.")
    else:
        logging.info(f"Uploading dataset ({file_size_mb:.2f} MB) to GCS...")

    # FIX: Tăng timeout lên 600 giây thay vì dùng thời gian mặc định của thư viện
    blob.upload_from_file(
        buf, 
        content_type="application/octet-stream",
        timeout=600
    )
    logging.info(f"Uploaded → gs://{BRONZE_BUCKET}/{gcs_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: Census ACS
# ══════════════════════════════════════════════════════════════════════════════
def fetch_acs_year(year: int) -> pd.DataFrame:
    core_vars  = ["B01001_001E"] + AGE_COLS
    bonus_vars = [
        "B19013_001E", "B17001_002E", "B17001_001E",
        "B27010_033E", "B27010_050E", "B27010_001E",
    ]
    var_str = ",".join(core_vars + bonus_vars)
    url = f"https://api.census.gov/data/{year}/acs/acs5?get={var_str}&for=county:*&in=state:*&key={CENSUS_KEY}"
    
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw[1:], columns=raw[0])
    df["county_fips"] = df["state"] + df["county"]
    df["census_year"] = year

    num_cols = ["B01001_001E"] + AGE_COLS + bonus_vars
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df = df.rename(columns={
        "B01001_001E": "total_population_raw",
        "B19013_001E": "median_household_income_raw",
        "B17001_002E": "pop_below_poverty_raw",
        "B17001_001E": "poverty_universe_raw",
        "B27010_033E": "uninsured_19_64_raw",
        "B27010_050E": "uninsured_65_plus_raw",
        "B27010_001E": "insurance_universe_raw",
    })
    df = df.rename(columns={col: f"age_raw_{col}" for col in AGE_COLS})

    keep = (
        ["county_fips", "census_year", "state", "county",
         "total_population_raw", "median_household_income_raw",
         "pop_below_poverty_raw", "poverty_universe_raw",
         "uninsured_19_64_raw", "uninsured_65_plus_raw", "insurance_universe_raw"]
        + [f"age_raw_{c}" for c in AGE_COLS]
    )
    return df[keep]

def collect_census_acs_bronze(exec_date: str) -> None:
    logging.info("[census_acs] Years: %s", ACS_YEARS)
    all_years = []
    for year in ACS_YEARS:
        try:
            logging.info(f"  → Fetching ACS {year}...")
            df_year = fetch_acs_year(year)
            all_years.append(df_year)
            logging.info(f"     {len(df_year)} counties")
        except Exception as e:
            logging.warning(f"  → ACS {year} failed (may not be released): {e}")

    if not all_years:
        raise RuntimeError("[census_acs] All ACS years failed.")

    df_all = pd.concat(all_years, ignore_index=True)
    upload_parquet_to_gcs(df_all, f"population/census_acs/{exec_date}/data.parquet")
    logging.info(f"[census_acs] Done. Total records: {len(df_all)}")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: HRSA AHRF (local files)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_column_names_from_sas(sas_file_path: str) -> list:
    """Helper: Parse headers từ file .sas metadata"""
    names = []
    with open(sas_file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        match = re.search(r'@(\d+)\s+([A-Za-z0-9_]+)\s+\$?\s*([0-9]+)\.\s*/\*(.*?)\*/', line)
        if match:
            description = match.group(4).strip()
            description = re.sub(r'\s+', '_', description)
            description = re.sub(r'[^A-Za-z0-9_]', '', description)
            names.append(description)

    # Handle duplicate columns
    seen = {}
    clean_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        clean_names.append(name)
    return clean_names

def _get_asc_colspecs(sas_file_path: str) -> tuple:
    """Helper: Parse colspecs và names cho file .asc từ metadata .sas"""
    colspecs, names = [], []
    with open(sas_file_path, 'r', encoding='latin1') as f:
        for line in f:
            match = re.search(r'@(\d+)\s+([A-Za-z0-9_]+)\s+\$?\s*([0-9]+)\.\s*/\*(.*?)\*/', line.strip())
            if match:
                start = int(match.group(1))
                width = int(match.group(3))
                description = match.group(4).strip()
                description = re.sub(r'\s+', '_', description)
                description = re.sub(r'[^A-Za-z0-9_]', '', description)
                
                colspecs.append((start - 1, start - 1 + width))
                names.append(description)

    # Handle duplicates
    seen, clean_names = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        clean_names.append(name)
        
    return colspecs, clean_names

def fetch_ahrf_all_vintages():
    """
    Generator yield dataframe theo từng năm để tránh OOM khi concat toàn bộ AHRF.
    Mỗi file AHRF có dung lượng lớn và hàng nghìn cột.
    """
    for config in AHRF_VINTAGES:
        vintage_year = config["data_year"]
        fmt = config["fmt"]
        file_path = config["data_file"]
        
        logging.info(f"[hrsa_ahrf] Processing year {vintage_year} format {fmt}")
        
        if not os.path.exists(file_path):
            logging.warning(f"[hrsa_ahrf] File not found: {file_path}. Skipping.")
            continue

        try:
            if fmt == "sas7bdat":
                sas_meta_path = file_path.replace(".sas7bdat", ".sas").replace("ahrf", "AHRF")
                if not os.path.exists(sas_meta_path):
                    # Fallback mapping if naming conventions vary
                    sas_meta_path = file_path.replace("ahrf2022.sas7bdat", "AHRF2021-2022.sas")
                
                df, _ = pyreadstat.read_sas7bdat(file_path)
                if os.path.exists(sas_meta_path):
                    real_headers = _extract_column_names_from_sas(sas_meta_path)
                    min_cols = min(len(df.columns), len(real_headers))
                    rename_dict = {old_col: real_headers[i] for i, old_col in enumerate(df.columns[:min_cols])}
                    df = df.rename(columns=rename_dict)

            elif fmt == "asc":
                # For 2021, 2022 ASC format
                sas_meta_path = file_path.replace(".asc", ".sas").replace("ahrf", "AHRF")
                colspecs, names = _get_asc_colspecs(sas_meta_path)
                df = pd.read_fwf(file_path, colspecs=colspecs, names=names, dtype=str, encoding='latin1')

            elif fmt == "csv":
                # For 2023, 2024, 2025 CSV format
                df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            
            else:
                raise ValueError(f"Unknown format {fmt}")

            # Đảm bảo có metadata year
            df["data_year"] = vintage_year
            yield vintage_year, df

        except Exception as e:
            logging.error(f"[hrsa_ahrf] Failed processing {vintage_year}: {e}", exc_info=True)
            raise

def collect_hrsa_ahrf_bronze(exec_date: str) -> None:
    logging.info("[hrsa_ahrf] Starting — loading %d vintages...", len(AHRF_VINTAGES))
    
    records_processed = 0
    for year, df_year in fetch_ahrf_all_vintages():
        custom_gcs_path = f"ahrf/hospital_resources/ahrf_{year}.parquet"
        
        upload_parquet_to_gcs(df_year, custom_gcs_path)
        records_processed += len(df_year)
        
    logging.info(f"[hrsa_ahrf] Done. Total records uploaded: {records_processed}")
    

def collect_hrsa_ahrf_bronze(exec_date: str) -> None:
    logging.info("[hrsa_ahrf] Starting — loading %d vintages...", len(AHRF_VINTAGES))
    df = fetch_ahrf_all_vintages()
    upload_parquet_to_gcs(df, f"population/hrsa_ahrf/{exec_date}/data.parquet")
    logging.info(f"[hrsa_ahrf] Done. Total records: {len(df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: CDC FluSurv-NET (local CSV)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_fluview_local() -> pd.DataFrame:
    if not os.path.exists(FLUVIEW_LOCAL):
        raise FileNotFoundError(f"[cdc_fluview] File not found: {FLUVIEW_LOCAL}")

    logging.info(f"[cdc_fluview] Loading: {FLUVIEW_LOCAL}")

    # FIX: Dynamically skip metadata/disclaimer blocks at top of CDC files
    header_skip = 0
    with open(FLUVIEW_LOCAL, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if "CATCHMENT" in line.upper() or "SURVEILLANCE" in line.upper() or "YEAR" in line.upper():
                header_skip = i
                raw_header = line.strip()
                break
        else:
            raise ValueError("[cdc_fluview] Could not locate valid header line in CSV.")

    raw_cols = [c.strip() for c in raw_header.split(",")]
    year_positions = [i for i, c in enumerate(raw_cols) if c.upper() == "YEAR"]

    if len(year_positions) == 2:
        raw_cols[year_positions[0]] = "SEASON_YEAR"
        raw_cols[year_positions[1]] = "MMWR_YEAR"
        logging.info("  → Duplicate YEAR detected: renamed to SEASON_YEAR, MMWR_YEAR")
    elif len(year_positions) == 1:
        raw_cols[year_positions[0]] = "MMWR_YEAR"

    normalized_cols = [c.strip().upper().replace(" ", "_") for c in raw_cols]

    # Load file cleanly starting right at the target dynamic layout block
    df = pd.read_csv(
        FLUVIEW_LOCAL,
        header=0,
        names=normalized_cols,
        dtype=str,
        encoding="utf-8-sig",
        skiprows=header_skip
    )

    logging.info(f"  → Raw rows: {len(df)} | Columns: {df.columns.tolist()}")

    filter_cols = {
        "AGE_CATEGORY": "Overall", "SEX_CATEGORY": "Overall",
        "RACE_CATEGORY": "Overall", "VIRUS_TYPE_CATEGORY": "Overall",
    }
    for col, val in filter_cols.items():
        if col in df.columns:
            df = df[df[col].str.strip().str.lower() == val.lower()]

    numeric_cols = [
        "MMWR_YEAR", "WEEK", "CUMULATIVE_RATE", "WEEKLY_RATE",
        "AGE_ADJUSTED_CUMULATIVE_RATE", "AGE_ADJUSTED_WEEKLY_RATE",
        "LOWER", "MEDIAN", "UPPER", "SEASON_YEAR"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    state_rows    = df[df["CATCHMENT"].str.strip() != "Entire Network"]
    national_rows = df[df["CATCHMENT"].str.strip() == "Entire Network"]
    logging.info(f"  → State-level rows: {len(state_rows)} | National rows: {len(national_rows)}")

    df["source"] = "CDC_FluSurv_NET_local"
    return df

def collect_fluview_bronze(exec_date: str) -> None:
    logging.info("[cdc_fluview] Starting...")
    df = fetch_fluview_local()
    upload_parquet_to_gcs(df, f"population/cdc_fluview/{exec_date}/data.parquet")
    logging.info(f"[cdc_fluview] Done. Records: {len(df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: USDA RUCC (local CSV)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_usda_rucc_local() -> pd.DataFrame:
    if not os.path.exists(RUCC_LOCAL):
        raise FileNotFoundError(f"[usda_rucc] File not found: {RUCC_LOCAL}")

    logging.info(f"[usda_rucc] Loading: {RUCC_LOCAL}")
    # FIX: Added encoding="latin-1" to resolve UnicodeDecodeError
    df_long = pd.read_csv(RUCC_LOCAL, dtype=str, encoding="latin-1")
    df_long.columns = df_long.columns.str.strip()

    df_wide = df_long.pivot_table(
        index=["FIPS", "State", "County_Name"],
        columns="Attribute",
        values="Value",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    rucc_col = next((c for c in df_wide.columns if c.upper().startswith("RUCC_") and c.upper() != "RUCC_DESCRIPTION"), None)
    pop_col  = next((c for c in df_wide.columns if "POPULATION" in c.upper()), None)
    desc_col = next((c for c in df_wide.columns if "DESCRIPTION" in c.upper()), None)

    if not rucc_col:
        raise ValueError(f"[usda_rucc] Cannot find RUCC code column. Available: {df_wide.columns.tolist()}")

    rename_map = {"FIPS": "county_fips", "State": "state_abbr", "County_Name": "county_name_usda", rucc_col: "rucc_code_raw"}
    if pop_col: rename_map[pop_col] = "reference_population"
    if desc_col: rename_map[desc_col] = "rucc_description"

    df_wide = df_wide.rename(columns=rename_map)
    df_wide["county_fips"] = df_wide["county_fips"].astype(str).str.zfill(5)
    df_wide["rucc_code_raw"] = pd.to_numeric(df_wide["rucc_code_raw"], errors="coerce").astype("Int64")

    if "reference_population" in df_wide.columns:
        df_wide["reference_population"] = pd.to_numeric(df_wide["reference_population"], errors="coerce").astype("Int64")

    df_wide = df_wide.dropna(subset=["county_fips", "rucc_code_raw"])
    df_wide = df_wide[df_wide["county_fips"].str.match(r"^\d{5}$")]
    df_wide["rucc_vintage"] = "2023"

    keep_cols = ["county_fips", "state_abbr", "county_name_usda", "rucc_code_raw", "rucc_vintage"]
    for opt in ["rucc_description", "reference_population"]:
        if opt in df_wide.columns: keep_cols.append(opt)

    return df_wide[keep_cols]

def collect_usda_rucc_bronze(exec_date: str) -> None:
    logging.info("[usda_rucc] Starting...")
    df = fetch_usda_rucc_local()
    upload_parquet_to_gcs(df, f"population/usda_rucc/{exec_date}/data.parquet")
    logging.info(f"[usda_rucc] Done. Counties: {len(df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: ILINet State Activity Indicator (local CSV)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ilinet_local() -> pd.DataFrame:
    if not os.path.exists(ILINET_LOCAL):
        raise FileNotFoundError(f"[ilinet] File not found: {ILINET_LOCAL}")

    logging.info(f"[ilinet] Loading: {ILINET_LOCAL}")
    # FIX: Changed sep="\t" to standard fallback comma sep="," because the logs show a standard comma format
    df = pd.read_csv(
        ILINET_LOCAL,
        sep=",",
        dtype=str,
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    
    required = {"STATENAME", "ACTIVITY_LEVEL", "ACTIVITY_LEVEL_LABEL", "WEEKEND", "WEEK", "SEASON"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[ilinet] Missing columns: {missing}. Available: {df.columns.tolist()}")

    df["season_start_year"] = df["SEASON"].str.split("-").str[0].pipe(pd.to_numeric, errors="coerce").astype("Int64")
    # FIX: Convert về microseconds thay vì để pandas default nanoseconds
    # Spark 2.1 không đọc được TIMESTAMP(NANOS) trong Parquet
    df["weekend_date"] = pd.to_datetime(
        df["WEEKEND"], format="%b-%d-%Y", errors="coerce"
    ).astype("datetime64[us]")  # [us] = microseconds, Spark-compatible
    
    df["week"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")
    df["activity_level_raw"] = pd.to_numeric(df["ACTIVITY_LEVEL"], errors="coerce").astype("Int64")
    df["state_name"] = df["STATENAME"].str.strip().str.title()
    df["source"] = "CDC_ILINet_local"

    return df[["state_name", "season_start_year", "SEASON", "week", "weekend_date", "activity_level_raw", "ACTIVITY_LEVEL_LABEL", "source"]]

def collect_ilinet_bronze(exec_date: str) -> None:
    logging.info("[ilinet] Starting...")
    df = fetch_ilinet_local()
    upload_parquet_to_gcs(df, f"population/cdc_ilinet/{exec_date}/data.parquet")
    logging.info(f"[ilinet] Done. Records: {len(df)}")


# ── Main Orchestration ────────────────────────────────────────────────────────
def collect_population_bronze(exec_date: str) -> None:
    logging.info("=== Population Bronze Collection Started ===")
    collectors = [
        ("census_acs", collect_census_acs_bronze),
        ("hrsa_ahrf",  collect_hrsa_ahrf_bronze),
        ("cdc_fluview", collect_fluview_bronze),
        ("cdc_ilinet",  collect_ilinet_bronze),
        ("usda_rucc",  collect_usda_rucc_bronze),
    ]

    errors = []
    for name, fn in collectors:
        try:
            fn(exec_date)
        except Exception as e:
            logging.error(f"[{name}] FAILED: {e}", exc_info=True)
            errors.append((name, str(e)))

    succeeded = [n for n, _ in collectors if n not in [e for e, _ in errors]]
    logging.info(
        f"\n=== Population Bronze Collection Finished ===\n"
        f"  Succeeded : {succeeded}\n"
        f"  Failed    : {[n for n, _ in errors] or 'None'}\n"
    )

    if errors:
        raise RuntimeError(f"Population collection partially failed. Failed sources: {[n for n, _ in errors]}. Details: {errors}")


if __name__ == "__main__":
    from datetime import datetime
    collect_population_bronze(datetime.now().strftime("%Y-%m-%d"))