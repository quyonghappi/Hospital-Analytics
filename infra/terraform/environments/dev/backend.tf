terraform {
  backend "gcs" {
    bucket = "project-8e2366a6-d3cc-40ee-9de-tf-state"
    prefix = "terraform/state/dev/data-platform"
  }
}