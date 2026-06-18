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


# ========================================================================
# 4. ML OBSERVABILITY DATASET (MLOPS LAYER)
# ========================================================================
resource "google_bigquery_dataset" "ml_observability" {
  dataset_id                  = "ml_observability_${var.env}"
  friendly_name               = "MLOps Monitoring & Evaluation (${var.env})"
  description                 = "Lưu trữ log dự báo của nhiều mô hình, SHAP chi tiết và actuals (ground truth) để monitor Model Drift."
  project                     = var.project_id
  location                    = var.region
  delete_contents_on_destroy  = var.env == "prod" ? false : var.force_destroy

  labels = merge(var.common_labels, {
    layer               = "mlops"
    data_classification = "internal"
  })
}

# ========================================================================
# 5. FEATURE STORE DATASET (PROCESSING / AI INPUT LAYER)
# ========================================================================
resource "google_bigquery_dataset" "hospital_feature_store" {
  dataset_id                  = "hospital_feature_store_${var.env}"
  friendly_name               = "Hospital Feature Store (${var.env})"
  description                 = "Nguồn cấp dữ liệu Input đã chuẩn hóa cho Vertex AI Training và Inference."
  project                     = var.project_id
  location                    = var.region
  delete_contents_on_destroy  = var.env == "prod" ? false : var.force_destroy

  labels = merge(var.common_labels, {
    layer               = "feature-store"
    data_classification = "confidential"
  })
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