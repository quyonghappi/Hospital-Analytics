# -*- coding: utf-8 -*-
"""
dim_geography Bronze Collector — 3 separate uploads, no merge at ingestion.

Sources → Bronze paths:
  Census ACS API   → geography/census_acs/{exec_date}/data.parquet
  Gazetteer local  → geography/gazetteer/{exec_date}/data.parquet
  Dartmouth HRR    → geography/dartmouth_hrr/{exec_date}/data.parquet

Merge happens downstream in Silver transformation (PySpark ETL).

DESIGN NOTE — Why geography is NOT collected per year:
  dim_geography is a static dimension. County names, state codes,
  HRR assignments, and land area do not change meaningfully across
  2020-2024. A single latest-year snapshot is sufficient.
  
  Contrast with dim_population (yearly): population counts,
  physician/nurse ratios change annually → per-year collection needed.

  Gazetteer strategy: load latest available year (2024 → 2023 fallback).
  area_km2 = land area, physically stable across years.
"""
import os
import io
import zipfile
import logging
import requests
import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ID    = os.getenv("GCP_PROJECT", "project-8e2366a6-d3cc-40ee-9de")
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "project-8e2366a6-d3cc-40ee-9de-bronze-raw-dev")
CENSUS_KEY    = os.getenv("CENSUS_API_KEY", "937589a4e18bf415305ccf41b6e8aa6ee0fcc258")
GAZ_DIR       = os.getenv("GAZ_DIR", r"C:\Users\HP\Documents\prj-hospital-analytics")

# Gazetteer: try latest → fallback in descending order
GAZ_YEAR_PRIORITY = [2024, 2023, 2022, 2021, 2020]

DARTMOUTH_FALLBACK_CHAIN = [
    ("2019", "https://data.dartmouthatlas.org/downloads/geography/ZipHsaHrr19.csv.zip", "csv"),
    ("2018", "https://data.dartmouthatlas.org/downloads/geography/ZipHsaHrr18.csv.zip", "csv"),
    ("2017", "https://data.dartmouthatlas.org/downloads/geography/ZipHsaHrr17.xls",     "xls"),
]

# ── GCS Helper ────────────────────────────────────────────────────────────────

def upload_parquet_to_gcs(df: pd.DataFrame, gcs_path: str) -> None:
    """In-memory serialize → GCS. No local tmp file."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    client = storage.Client(project=PROJECT_ID)
    client.bucket(BRONZE_BUCKET).blob(gcs_path).upload_from_file(
        buf, content_type="application/octet-stream"
    )
    logging.info(f"Uploaded → gs://{BRONZE_BUCKET}/{gcs_path}")


# ── Source 1: Census ACS — county_name, state_code, state_name ───────────────

def fetch_county_names() -> pd.DataFrame:
    """
    ACS 5-Year: NAME field at county level. Single call (2023 vintage).
    Raw columns only — Silver handles suffix stripping and normalization.
    """
    url = (
        f"https://api.census.gov/data/2023/acs/acs5"
        f"?get=NAME&for=county:*&in=state:*&key={CENSUS_KEY}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw[1:], columns=raw[0])
    df["county_fips"] = df["state"] + df["county"]

    # Minimal parse only — "Los Angeles County, California"
    # Silver will strip suffix (County/Parish/Borough...)
    df[["county_name", "state_name"]] = (
        df["NAME"].str.rsplit(", ", n=1, expand=True)
    )
    df["state_code"] = df["state"].map(_fips_state_map())

    # acs_vintage: audit which ACS year this snapshot came from
    df["acs_vintage"] = 2023

    return df[["county_fips", "county_name", "state_name", "state_code",
               "NAME", "acs_vintage"]]


def _fips_state_map() -> dict:
    return {
        "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
        "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
        "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
        "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
        "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
        "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
        "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
        "55":"WI","56":"WY","72":"PR",
    }


# ── Source 2: Gazetteer local — SINGLE latest year ────────────────────────────

def load_gazetteer_latest() -> pd.DataFrame:
    """
    Load Gazetteer từ local files — single latest available year.
    
    WHY single year:
      area_km2 = land area, does not change across 2020-2024.
      Stacking all years (previous approach) creates 5 rows/county
      → incorrect fan-out when Silver joins on county_fips.

    Fallback: iterate GAZ_YEAR_PRIORITY until a file is found.
    """
    SQ_MI_TO_KM2 = 2.58999

    loaded_year = None
    df = None

    for year in GAZ_YEAR_PRIORITY:
        file_path = os.path.join(GAZ_DIR, f"{year}_Gaz_counties_national.txt")
        if os.path.exists(file_path):
            logging.info(f"[gazetteer] Loading year {year}: {file_path}")
            df = pd.read_csv(file_path, sep="\t", dtype={"GEOID": str})
            loaded_year = year
            break
        else:
            logging.warning(f"[gazetteer] File not found for year {year}, trying next...")

    if df is None:
        raise FileNotFoundError(
            f"[gazetteer] No Gazetteer file found for any year in {GAZ_YEAR_PRIORITY}. "
            f"Expected path pattern: {GAZ_DIR}/<year>_Gaz_counties_national.txt"
        )

    df["county_fips"] = df["GEOID"].str.zfill(5)
    df["area_sqmi"]   = pd.to_numeric(df["ALAND_SQMI"], errors="coerce")
    df["area_km2"]    = df["area_sqmi"] * SQ_MI_TO_KM2
    df["gaz_vintage"] = loaded_year  # audit: which year file was used

    result = df[["county_fips", "area_sqmi", "area_km2", "gaz_vintage"]]

    # Sanity check: no duplicate county_fips
    dupes = result["county_fips"].duplicated().sum()
    if dupes > 0:
        logging.warning(f"[gazetteer] {dupes} duplicate county_fips found — check source file.")

    logging.info(
        f"[gazetteer] Loaded {len(result)} counties "
        f"| Vintage: {loaded_year} "
        f"| area_km2 null: {result['area_km2'].isna().sum()}"
    )
    return result


# ── Source 3: Dartmouth Atlas — hrr_region ───────────────────────────────────
def fetch_hrr_region() -> pd.DataFrame:
    """
    Dartmouth Atlas ZIP→HRR crosswalk.
    Fallback chain: 2019 csv.zip → 2018 csv.zip → 2017 xls.

    Plurality vote: assign county → HRR by dominant ZIP area coverage.
    Static mapping — HRR boundaries stable, no per-year collection needed.
    """
    df_hrr      = None
    used_vintage = None

    for vintage, url, fmt in DARTMOUTH_FALLBACK_CHAIN:
        try:
            logging.info(f"[dartmouth_hrr] Attempting {vintage} ({fmt}): {url}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()

            if fmt == "csv":
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                    logging.info(f"  → Extracting: {csv_name}")
                    with zf.open(csv_name) as f:
                        df_hrr = pd.read_csv(f, dtype=str)
            else:
                df_hrr = pd.read_excel(
                    io.BytesIO(resp.content),
                    dtype=str,
                    engine="xlrd",
                )

            used_vintage = vintage
            logging.info(f"  → Loaded {vintage} vintage. Rows: {len(df_hrr)}")
            break

        except Exception as e:
            logging.warning(f"[dartmouth_hrr] {vintage} failed: {e}. Trying next...")
            continue

    if df_hrr is None:
        raise RuntimeError("[dartmouth_hrr] All Dartmouth Atlas download attempts failed.")

    # Normalize columns — handle known alias differences across vintages
    df_hrr.columns = df_hrr.columns.str.lower().str.strip()
    col_aliases = {"zipcode19": "zipcode", "zipcode18": "zipcode"}
    df_hrr.rename(columns=col_aliases, inplace=True)

    required = {"zipcode", "hrrnum", "hrrstate", "hrrcity"}
    missing  = required - set(df_hrr.columns)
    if missing:
        raise ValueError(
            f"[dartmouth_hrr] Schema mismatch. Missing: {missing}. "
            f"Available: {df_hrr.columns.tolist()}"
        )

    df_hrr["zip5"]        = df_hrr["zipcode"].astype(str).str.zfill(5).str[:5]
    df_hrr["hrr_region"]  = (
        df_hrr["hrrstate"].str.strip() + "-" + df_hrr["hrrcity"].str.strip()
    )
    df_hrr["hrr_vintage"] = used_vintage
    df_hrr = df_hrr[["zip5", "hrr_region", "hrrnum", "hrr_vintage"]].drop_duplicates("zip5")

    # ZCTA → County crosswalk (local file)
    zcta_path = os.path.join(GAZ_DIR, "tab20_zcta520_county20_natl.txt")
    if not os.path.exists(zcta_path):
        raise FileNotFoundError(
            f"[dartmouth_hrr] Missing ZCTA crosswalk: {zcta_path}\n"
            f"Download from: https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
            f"zcta520/tab20_zcta520_county20_natl.txt"
        )

    logging.info(f"[dartmouth_hrr] Loading ZCTA crosswalk: {zcta_path}")
    df_zcta = pd.read_csv(
        zcta_path,
        sep="|",
        dtype={"GEOID_ZCTA5_20": str, "GEOID_COUNTY_20": str},
        usecols=["GEOID_ZCTA5_20", "GEOID_COUNTY_20", "AREALAND_PART"],
    )
    df_zcta.columns = ["zip5", "county_fips", "area_pct"]

    # Plurality vote: per county, pick HRR with max cumulative area coverage
    df_county_hrr = (
        df_zcta.merge(df_hrr, on="zip5", how="left")
        .dropna(subset=["hrr_region"])
        .groupby(["county_fips", "hrr_region", "hrrnum", "hrr_vintage"])["area_pct"]
        .sum()
        .reset_index()
        .sort_values("area_pct", ascending=False)
        .drop_duplicates("county_fips")
        [["county_fips", "hrr_region", "hrrnum", "hrr_vintage"]]
    )

    total_counties  = df_zcta["county_fips"].nunique()
    mapped_counties = df_county_hrr["county_fips"].nunique()
    unmapped        = total_counties - mapped_counties
    logging.info(
        f"[dartmouth_hrr] Mapped: {mapped_counties}/{total_counties} counties "
        f"| Unmapped: {unmapped} | Vintage: {used_vintage}"
    )
    if unmapped / total_counties > 0.05:
        logging.warning(
            f"[dartmouth_hrr] Unmapped rate {unmapped/total_counties:.1%} > 5% threshold."
        )

    return df_county_hrr


# ── Orchestration ─────────────────────────────────────────────────────────────

def collect_geography_bronze(exec_date: str) -> None:
    """
    3 sources → 3 independent Bronze uploads.
    Each collector is static (no year loop) — see module docstring.
    """
    logging.info("=== Geography Bronze Collection Started ===")
    errors = []

    # Source 1: Census ACS
    try:
        logging.info("[1/3] Census ACS county names (2023 vintage)...")
        df_acs = fetch_county_names()
        logging.info(f"  → {len(df_acs)} counties")
        upload_parquet_to_gcs(
            df_acs,
            f"geography/census_acs/{exec_date}/data.parquet"
        )
    except Exception as e:
        logging.error(f"[census_acs] FAILED: {e}", exc_info=True)
        errors.append(("census_acs", str(e)))

    # Source 2: Gazetteer (single latest year)
    try:
        logging.info("[2/3] Gazetteer (latest available year)...")
        df_gaz = load_gazetteer_latest()
        upload_parquet_to_gcs(
            df_gaz,
            f"geography/gazetteer/{exec_date}/data.parquet"
        )
    except Exception as e:
        logging.error(f"[gazetteer] FAILED: {e}", exc_info=True)
        errors.append(("gazetteer", str(e)))

    # Source 3: Dartmouth HRR
    try:
        logging.info("[3/3] Dartmouth Atlas HRR regions...")
        df_hrr = fetch_hrr_region()
        logging.info(f"  → {len(df_hrr)} county-HRR mappings")
        upload_parquet_to_gcs(
            df_hrr,
            f"geography/dartmouth_hrr/{exec_date}/data.parquet"
        )
    except Exception as e:
        logging.error(f"[dartmouth_hrr] FAILED: {e}", exc_info=True)
        errors.append(("dartmouth_hrr", str(e)))

    # Final status
    if errors:
        raise RuntimeError(
            f"Geography collection partially failed. "
            f"Failed: {[s for s, _ in errors]}. Details: {errors}"
        )

    logging.info(
        "=== Geography Bronze Collection Finished ===\n"
        f"  gs://{BRONZE_BUCKET}/geography/census_acs/{exec_date}/data.parquet\n"
        f"  gs://{BRONZE_BUCKET}/geography/gazetteer/{exec_date}/data.parquet\n"
        f"  gs://{BRONZE_BUCKET}/geography/dartmouth_hrr/{exec_date}/data.parquet"
    )


if __name__ == "__main__":
    from datetime import datetime
    collect_geography_bronze(datetime.now().strftime("%Y-%m-%d"))