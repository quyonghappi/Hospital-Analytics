resource "google_bigquery_dataset" "hospital_dwh" {
  dataset_id                 = "hospital_dwh_${var.env}"
  project                    = var.project_id
  location                   = var.region
  delete_contents_on_destroy = var.force_destroy
  labels                     = var.common_labels
}