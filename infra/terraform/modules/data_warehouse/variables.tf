variable "project_id" { type = string }
variable "region" { type = string }
variable "env" { type = string }
variable "force_destroy" { type = bool }
variable "common_labels" { type = map(string) }
variable "silver_bucket_name" {
  description = "Tên của bucket Silver truyền từ module storage sang"
  type        = string
}