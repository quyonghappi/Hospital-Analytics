resource "google_bigquery_dataset" "hospital_dwh" {
  dataset_id                 = "hospital_dwh_${var.env}"
  location                   = var.region
  delete_contents_on_destroy = var.force_destroy
}

# Cấp quyền cho Dataform tạo/sửa bảng trong DWH
resource "google_bigquery_dataset_iam_member" "dataform_editor" {
  dataset_id = google_bigquery_dataset.hospital_dwh.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${var.dataform_sa_email}"
}