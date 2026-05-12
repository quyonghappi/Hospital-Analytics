# Orchestrator
module "security" {
  source     = "../../modules/security"
  project_id = var.project_id
  env        = var.env
}

module "storage" {
  source          = "../../modules/storage"
  project_id      = var.project_id
  region          = var.region
  env             = var.env
  prevent_destroy = false # DEV cho phép xóa
  force_destroy   = true  # DEV cho phép xóa bucket dù có data
  spark_sa_email  = module.security.spark_sa_email
}

module "data_warehouse" {
  source            = "../../modules/data_warehouse"
  project_id        = var.project_id
  region            = var.region
  env               = var.env
  force_destroy     = true
  dataform_sa_email = module.security.dataform_sa_email
}