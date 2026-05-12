# Quarantine và Silver zone theo spec
locals {
  buckets = ["bronze-raw", "silver-curated", "quarantine"]
}

resource "google_storage_bucket" "data_lake" {
  for_each      = toset(local.buckets)
  name          = "${var.project_id}-${each.key}-${var.env}"
  location      = var.region
  force_destroy = var.force_destroy # Quyết định bởi environment

  lifecycle {
    prevent_destroy = var.prevent_destroy
  }
}

# Gán quyền đọc/ghi Bronze cho Spark SA
resource "google_storage_bucket_iam_member" "spark_bronze_admin" {
  bucket = google_storage_bucket.data_lake["bronze-raw"].name
  role   = "roles/storage.objectAdmin" # Spark cần ghi/sửa, ObjectViewer là không đủ nếu có job đẩy data
  member = "serviceAccount:${var.spark_sa_email}"
}