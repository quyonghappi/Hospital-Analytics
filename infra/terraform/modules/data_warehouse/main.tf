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