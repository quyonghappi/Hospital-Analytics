# Quarantine và Silver zone theo spec
locals {
  buckets = ["bronze-raw", "silver-curated", "quarantine"]
}

resource "google_storage_bucket" "data_lake" {
  for_each      = toset(local.buckets)
  name          = "${var.project_id}-${each.key}-${var.env}"
  location      = var.region
  force_destroy = var.force_destroy # Quyết định bởi environment
  labels        = var.common_labels
  uniform_bucket_level_access = true

  dynamic "lifecycle_rule" {
    for_each = each.key == "bronze-raw" ? [1] : []
    content {
      condition {
        age = 30
      }
      action {
        type          = "SetStorageClass"
        storage_class = "COLDLINE"
      }
    }
  }
}