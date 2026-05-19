# Cấu hình danh sách các bảng Staging (External Tables trỏ ra GCS Silver)
locals {
  silver_tables = {
    # ---------------------------------------------------------
    # BẢNG FACT (PARTITIONED)
    # ---------------------------------------------------------
    # Delete from locals block, assign init task to Airflow DAG 
    
    # ---------------------------------------------------------
    # BẢNG DIMENSION (UNPARTITIONED)
    # ---------------------------------------------------------
    "ext_dim_hospital" = {
      gcs_prefix               = "dim_hospital"
      hive_partitioned         = false
      require_partition_filter = false
      columns = [
        { name = "hospital_id", type = "STRING", description = "Mã chứng nhận bệnh viện (CMS Provider CCN)" },
        { name = "hospital_name", type = "STRING", description = "Tên đầy đủ cơ sở y tế" },
        { name = "hospital_type", type = "STRING", description = "Phân loại bệnh viện" },
        { name = "ownership_type", type = "STRING", description = "Loại hình sở hữu" },
        { name = "rural_urban_flag", type = "STRING", description = "Phân loại khu vực Nông thôn hay Thành thị" },
        { name = "staffed_beds_capacity", type = "INT64", description = "Trần năng lực giường bệnh thiết kế" },
        { name = "icu_beds_capacity", type = "INT64", description = "Trần năng lực giường ICU thiết kế" },
        { name = "total_fte_staff", type = "FLOAT64", description = "Tổng số nhân viên quy đổi toàn thời gian" },
        { name = "state", type = "STRING", description = "Mã bang" },
        { name = "county_fips", type = "STRING", description = "Mã định danh hạt (FIPS)" },
        { name = "latitude", type = "FLOAT64", description = "Vĩ độ" },
        { name = "longitude", type = "FLOAT64", description = "Kinh độ" }
      ]
    }

    "ext_dim_geography" = {
      gcs_prefix               = "dim_geography"
      hive_partitioned         = false
      require_partition_filter = false
      columns = [
        { name = "county_fips", type = "STRING", description = "Mã định danh hạt (County FIPS code)" },
        { name = "county_name", type = "STRING", description = "Tên hạt/quận" },
        { name = "state_code", type = "STRING", description = "Mã bang" },
        { name = "state_name", type = "STRING", description = "Tên bang đầy đủ" },
        { name = "hrr_region", type = "STRING", description = "Khu vực chuyển tuyến bệnh viện" },
        { name = "area_km2", type = "FLOAT64", description = "Diện tích của hạt/quận" }
      ]
    }

    "ext_dim_population" = {
      gcs_prefix               = "dim_population"
      hive_partitioned         = false
      require_partition_filter = false
      columns = [
        { name = "county_fips", type = "STRING", description = "Mã định danh hạt (County FIPS code)" },
        { name = "census_year", type = "INT64", description = "Năm thống kê dân số" },
        { name = "total_population", type = "INT64", description = "Tổng dân số trong khu vực" },
        { name = "pop_65_plus_pct", type = "FLOAT64", description = "Tỷ lệ phần trăm dân số trên 65 tuổi" },
        { name = "population_density", type = "FLOAT64", description = "Mật độ dân số" },
        { name = "physicians_per_100k", type = "FLOAT64", description = "Số lượng bác sĩ/100K dân" },
        { name = "nurses_per_100k", type = "FLOAT64", description = "Số lượng y tá/điều dưỡng/100K dân" }
      ]
    }

    "ext_dim_date" = {
      gcs_prefix               = "dim_date"
      hive_partitioned         = false
      require_partition_filter = false
      columns = [
        { name = "full_date", type = "DATE", description = "Ngày đầy đủ" },
        { name = "day_of_week", type = "STRING", description = "Thứ trong tuần" },
        { name = "week_number", type = "INT64", description = "Số thứ tự tuần trong năm" },
        { name = "month", type = "INT64", description = "Tháng" },
        { name = "quarter", type = "INT64", description = "Quý" },
        { name = "year", type = "INT64", description = "Năm" },
        { name = "season", type = "STRING", description = "Mùa trong năm" },
        { name = "is_holiday", type = "BOOLEAN", description = "Cờ báo hiệu ngày lễ" }
      ]
    }
  }
}