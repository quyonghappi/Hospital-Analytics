# ==========================================
# BATCH 1: CORE DATA PLATFORM SAs
# ==========================================
resource "google_service_account" "composer_orch" {
  account_id   = "sa-composer-${var.env}"
  display_name = "Airflow Orchestrator SA (${var.env})"
}

resource "google_service_account" "spark_processing" {
  account_id   = "sa-spark-${var.env}"
  display_name = "Dataproc Spa
  
  rk SA (${var.env})"
}

resource "google_service_account" "dataform_transform" {
  account_id   = "sa-dataform-${var.env}"
  display_name = "Dataform BQ Transform SA (${var.env})"
}

# for quota bypass
resource "time_sleep" "wait_60_seconds" {
  depends_on = [
    google_service_account.composer_orch,
    google_service_account.spark_processing,
    google_service_account.dataform_transform
  ]
  create_duration = "60s"
}

# ==========================================
# BATCH 2: DOWNSTREAM SAs (ML & SERVING)
# ==========================================
resource "google_service_account" "vertex_ml" {
  depends_on   = [time_sleep.wait_60_seconds]
  account_id   = "sa-vertex-${var.env}"
  display_name = "Vertex AI ML SA (${var.env})"
}

resource "google_service_account" "looker_serving" {
  depends_on   = [time_sleep.wait_60_seconds]
  account_id   = "sa-looker-${var.env}"
  display_name = "Looker Studio BI SA (${var.env})"
}

resource "google_service_account" "alert_engine" {
  depends_on   = [time_sleep.wait_60_seconds]
  account_id   = "sa-alert-${var.env}"
  display_name = "Cloud Functions Alert SA (${var.env})"
}

# ==========================================
# IAM BINDINGS
# ==========================================
resource "google_project_iam_member" "spark_bq_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.spark_processing.email}"
}

# Impersonation
resource "google_service_account_iam_member" "composer_impersonate_spark_user" {
  service_account_id = google_service_account.spark_processing.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.composer_orch.email}"
}

resource "google_service_account_iam_member" "composer_impersonate_spark_token" {
  service_account_id = google_service_account.spark_processing.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.composer_orch.email}"
}

# ==========================================
# OUTPUTS
# ==========================================
output "spark_sa_email" { value = google_service_account.spark_processing.email }
output "dataform_sa_email" { value = google_service_account.dataform_transform.email }
output "vertex_sa_email" { value = google_service_account.vertex_ml.email }
output "looker_sa_email" { value = google_service_account.looker_serving.email }