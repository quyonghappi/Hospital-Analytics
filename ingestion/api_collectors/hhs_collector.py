def main(execution_date):
    # execution_date là string định dạng YYYY-MM-DD từ Airflow
    report_date = datetime.strptime(execution_date, "%Y-%m-%d")
    
    # ... logic fetch data ...
    
    upload_to_bronze(df, source_name="hhs_capacity", report_date=report_date, schema=HHS_SCHEMA)