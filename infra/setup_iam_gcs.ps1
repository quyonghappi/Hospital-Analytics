$PROJECT_ID = "project-8e2366a6-d3cc-40ee-9de"
$REGION = "asia-southeast1"

# Tạo hệ thống Buckets 
gcloud storage buckets create gs://$PROJECT_ID-bronze-raw --location=$REGION
gcloud storage buckets create gs://$PROJECT_ID-silver-validated --location=$REGION
gcloud storage buckets create gs://$PROJECT_ID-spark-temp --location=$REGION
gcloud storage buckets create gs://$PROJECT_ID-ml-artifacts --location=$REGION

# Tạo BigQuery Datasets
bq mk --location=$REGION --dataset ${PROJECT_ID}:hospital_dwh
bq mk --location=$REGION --dataset ${PROJECT_ID}:ml_predictions

gcloud iam service-accounts create sa-composer-orch --display-name="Airflow Orchestrator SA"
gcloud iam service-accounts create sa-spark-processing --display-name="Dataproc Spark SA"
gcloud iam service-accounts create sa-dataform-transform --display-name="Dataform BQ Transform SA"
gcloud iam service-accounts create sa-vertex-ml --display-name="Vertex AI Pipeline SA"
gcloud iam service-accounts create sa-looker-serving --display-name="Looker Studio BI SA"
gcloud iam service-accounts create sa-alert-engine --display-name="Cloud Functions Alert SA"

# Gán biến email để dùng cho bước cấp quyền
$SA_COMPOSER = "sa-composer-orch@${PROJECT_ID}.iam.gserviceaccount.com"
$SA_SPARK = "sa-spark-processing@${PROJECT_ID}.iam.gserviceaccount.com"
$SA_DATAFORM = "sa-dataform-transform@${PROJECT_ID}.iam.gserviceaccount.com"
$SA_VERTEX = "sa-vertex-ml@${PROJECT_ID}.iam.gserviceaccount.com"
$SA_LOOKER = "sa-looker-serving@${PROJECT_ID}.iam.gserviceaccount.com"
$SA_ALERT = "sa-alert-engine@${PROJECT_ID}.iam.gserviceaccount.com"

# --- 1. SPARK SA (GCS + BQ) ---
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-bronze-raw --member="serviceAccount:$SA_SPARK" --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-silver-validated --member="serviceAccount:$SA_SPARK" --role="roles/storage.objectAdmin"
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-spark-temp --member="serviceAccount:$SA_SPARK" --role="roles/storage.objectAdmin"
gcloud bigquery datasets add-iam-policy-binding hospital_dwh --member="serviceAccount:$SA_SPARK" --role="roles/bigquery.dataEditor"

# --- 2. VERTEX SA (GCS + BQ + Vertex Compute) ---
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-ml-artifacts --member="serviceAccount:$SA_VERTEX" --role="roles/storage.objectAdmin"
gcloud bigquery datasets add-iam-policy-binding hospital_dwh --member="serviceAccount:$SA_VERTEX" --role="roles/bigquery.dataViewer"
gcloud bigquery datasets add-iam-policy-binding ml_predictions --member="serviceAccount:$SA_VERTEX" --role="roles/bigquery.dataEditor"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_VERTEX" --role="roles/bigquery.jobUser"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_VERTEX" --role="roles/aiplatform.customCodeServiceAgent"

# --- 3. COMPOSER SA (Orchestration & Impersonation) ---
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_COMPOSER" --role="roles/composer.worker"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_COMPOSER" --role="roles/dataproc.developer"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_COMPOSER" --role="roles/aiplatform.user"

# Cấp quyền cho Composer SA được phép "nhập vai" các SA khác
gcloud iam service-accounts add-iam-policy-binding $SA_SPARK --member="serviceAccount:$SA_COMPOSER" --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts add-iam-policy-binding $SA_VERTEX --member="serviceAccount:$SA_COMPOSER" --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts add-iam-policy-binding $SA_DATAFORM --member="serviceAccount:$SA_COMPOSER" --role="roles/iam.serviceAccountUser"