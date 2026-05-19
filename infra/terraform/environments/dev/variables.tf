variable "project_id" { type = string }
variable "region" { type = string }
variable "env" {
  type    = string
  default = "dev"
}
variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
}