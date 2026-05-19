# Orchestrator
provider "google" {
  project = var.project_id
  region  = var.region
}

module "storage" {
  source        = "../../modules/storage"
  project_id    = var.project_id
  region        = var.region
  env           = var.env
  force_destroy = true # DEV cho phép xóa bucket dù có data
  # spark_sa_email  = module.security.spark_sa_email
  common_labels = var.common_labels
}

module "data_warehouse" {
  source        = "../../modules/data_warehouse"
  project_id    = var.project_id
  region        = var.region
  env           = var.env
  force_destroy = true
  # dataform_sa_email = module.security.dataform_sa_email
  common_labels      = var.common_labels
  silver_bucket_name = module.storage.silver_bucket_name
}