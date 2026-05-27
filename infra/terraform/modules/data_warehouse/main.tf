# ------------------------------------------------------------------------
# 1. SERVING DATASET (GOLD ZONE - DWH)
# ------------------------------------------------------------------------
resource "google_bigquery_dataset" "hospital_dwh" {
  dataset_id                 = "hospital_dwh_${var.env}"
  friendly_name              = "Hospital DWH Gold"
  description                = "Curated dataset for Hospital Utilization. Accessed by Looker Studio and Vertex AI."
  project                    = var.project_id
  location                   = var.region
  delete_contents_on_destroy = var.env == "prod" ? false : var.force_destroy
  
  labels = merge(var.common_labels, {
    layer              = "gold"
    data_classification = "confidential"
  })
}

# ------------------------------------------------------------------------
# 2. STAGING DATASET (SILVER ZONE EXTERNAL TABLES)
# ------------------------------------------------------------------------
resource "google_bigquery_dataset" "hospital_staging" {
  dataset_id                 = "hospital_staging_${var.env}"
  friendly_name              = "Hospital Staging Silver"
  description                = "External tables pointing to GCS Silver Zone. Temporary processing layer."
  project                    = var.project_id
  location                   = var.region
  
  delete_contents_on_destroy = var.env == "prod" ? false : var.force_destroy

  default_table_expiration_ms = 604800000 * 3

  labels = merge(var.common_labels, {
    layer = "silver-staging"
  })
}

# ------------------------------------------------------------------------
# 3. ML PREDICTIONS DATASET (AI LAYER)
# ------------------------------------------------------------------------
resource "google_bigquery_dataset" "ml_predictions" {
  dataset_id                  = "ml_predictions_${var.env}"
  friendly_name               = "Hospital ML Forecasts (${var.env})"
  description                 = "Lưu trữ dự báo 7 ngày tới, SHAP values và alert flags từ Vertex AI Batch Prediction."
  project                     = var.project_id
  location                    = var.region
  delete_contents_on_destroy  = var.env == "prod" ? false : var.force_destroy

  labels = merge(var.common_labels, {
    layer               = "ai-ml"
    data_classification = "confidential"
  })
}

# ------------------------------------------------------------------------
# 4. ML FORECAST RESULTS TABLE (PRE-DEFINED SCHEMA & PARTITIONING)
# ------------------------------------------------------------------------
resource "google_bigquery_table" "ml_forecast_results" {
  dataset_id          = google_bigquery_dataset.ml_predictions.dataset_id
  table_id            = "ml_forecast_results"
  deletion_protection = var.env == "prod" ? true : false

  # Partition theo ngày dự báo để Looker Studio query rẻ và nhanh
  time_partitioning {
    type                     = "DAY"
    field                    = "forecast_date"
    require_partition_filter = true
  }

  # Cluster theo hospital_id để tối ưu UI filter (khi user chọn 1 bệnh viện trên bản đồ)
  clustering = ["hospital_id"]

  schema = <<EOF
[
  {"name": "forecast_id", "type": "STRING", "mode": "REQUIRED", "description": "UUID của dòng dự báo"},
  {"name": "hospital_id", "type": "STRING", "mode": "REQUIRED", "description": "Khóa ngoại nối với dim_hospital"},
  {"name": "forecast_date", "type": "DATE", "mode": "REQUIRED", "description": "Ngày mà kết quả dự báo hướng tới"},
  {"name": "run_date", "type": "DATE", "mode": "REQUIRED", "description": "Ngày chạy mô hình sinh ra dự báo"},
  {"name": "predicted_occupancy_rate", "type": "FLOAT64", "mode": "NULLABLE", "description": "Tỷ lệ lấp đầy dự báo"},
  {"name": "predicted_patient_volume", "type": "INT64", "mode": "NULLABLE", "description": "Lưu lượng bệnh nhân dự báo"},
  {"name": "predicted_los", "type": "FLOAT64", "mode": "NULLABLE", "description": "Thời gian lưu trú trung bình dự kiến"},
  {"name": "confidence_lower_95", "type": "FLOAT64", "mode": "NULLABLE", "description": "Giới hạn dưới của khoảng tin cậy 95%"},
  {"name": "confidence_upper_95", "type": "FLOAT64", "mode": "NULLABLE", "description": "Giới hạn trên của khoảng tin cậy 95%"},
  {"name": "shap_feature_1", "type": "STRING", "mode": "NULLABLE", "description": "Yếu tố ảnh hưởng mạnh nhất 1"},
  {"name": "shap_feature_2", "type": "STRING", "mode": "NULLABLE", "description": "Yếu tố ảnh hưởng mạnh nhất 2"},
  {"name": "shap_feature_3", "type": "STRING", "mode": "NULLABLE", "description": "Yếu tố ảnh hưởng mạnh nhất 3"},
  {"name": "model_version", "type": "STRING", "mode": "NULLABLE", "description": "Phiên bản của model Vertex AI"},
  {"name": "alert_flag", "type": "BOOLEAN", "mode": "NULLABLE", "description": "Cờ cảnh báo sớm nếu dự báo quá tải (>90%)"}
]
EOF
}

# Khởi tạo Assertion Dataset với cơ chế Auto-Delete
resource "google_bigquery_dataset" "assertions" {
  dataset_id                  = "data_assertions_${var.env}"
  friendly_name               = "Hospital Dataform Assertions (${var.env})"
  description                 = "Lưu trữ kết quả Data Quality rules của Dataform. Tự động xóa sau 7 ngày."
  location                    = var.region
  
  # CRITICAL COST OPTIMIZATION: Bảng assertion tự động bốc hơi sau 7 ngày (604800000 ms)
  default_table_expiration_ms = 604800000 
  
  labels = {
    environment = var.env
    layer       = "data-quality"
  }
}

# Tạo External Tables
resource "google_bigquery_table" "silver_external_tables" {
  for_each   = local.silver_tables
  dataset_id = google_bigquery_dataset.hospital_staging.dataset_id
  table_id   = each.key

  # FIX: External/staging tables không cần protection
  deletion_protection = false
  
   lifecycle {
    # Tránh recreate khi chỉ thay đổi description hoặc label
    ignore_changes = [description, labels]
  }


  # Định nghĩa schema thông qua jsonencode
  schema = length(each.value.columns) > 0 ? jsonencode([
    for col in each.value.columns : {
      name        = col.name
      type        = col.type
      mode        = "NULLABLE"
    }
  ]) : null

  external_data_configuration {
    autodetect    = length(each.value.columns) > 0 ? false : true
    source_format = "PARQUET"
    source_uris   = [
      "gs://${var.project_id}-silver-curated-${var.env}/${each.value.gcs_prefix}/*"
    ]

    dynamic "hive_partitioning_options" {
      for_each = each.value.hive_partitioned ? [1] : []
      content {
        mode                     = "STRINGS"
        source_uri_prefix        = "gs://${var.project_id}-silver-curated-${var.env}/${each.value.gcs_prefix}"
        require_partition_filter = each.value.require_partition_filter
      }
    }
  }
}