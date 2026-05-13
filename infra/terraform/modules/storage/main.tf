locals {
  buckets = {
    "bronze-raw"     = { storage_class = "STANDARD", nearline_days = 30,   archive_days = 365 }
    "silver-curated" = { storage_class = "STANDARD", nearline_days = 14,   archive_days = 90 }
    "quarantine"     = { storage_class = "NEARLINE", nearline_days = null, archive_days = 90 }
  }
}

resource "google_storage_bucket" "data_lake" {
  for_each      = local.buckets
  
  name          = "${var.project_id}-${each.key}-${var.env}"
  location      = var.region

  storage_class = each.value.storage_class 
  
  force_destroy = var.force_destroy
  labels        = var.common_labels
  uniform_bucket_level_access = true

  versioning {
    enabled = true 
  }

  # Lifecycle 1
  dynamic "lifecycle_rule" {
    for_each = each.value.nearline_days != null ? [1] : []
    content {
      condition {
        age = each.value.nearline_days
      }
      action {
        type          = "SetStorageClass"
        storage_class = "NEARLINE"
      }
    }
  }

  # Lifecycle 2
  dynamic "lifecycle_rule" {
    for_each = each.value.archive_days != null ? [1] : []
    content {
      condition {
        age = each.value.archive_days
      }
      action {
        type          = "SetStorageClass"
        storage_class = "ARCHIVE"
      }
    }
  }
  
  # Xóa các non-current versions sau 7 ngày
  lifecycle_rule {
    condition {
      num_newer_versions = 1
      days_since_noncurrent_time = 7
    }
    action {
      type = "Delete"
    }
  }
}

output "silver_bucket_name" {
  description = "Tên bucket Silver được sinh ra từ vòng lặp for_each"
  value       = google_storage_bucket.data_lake["silver-curated"].name
}