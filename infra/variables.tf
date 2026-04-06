variable "git_sha" {
  description = "Git commit SHA — used to identify the model pushed to S3"
  type        = string
}

variable "datasets_bucket" {
  description = "S3 bucket for DVC dataset storage"
  type        = string
  default     = "mlsecops-datasets-351611731527"
}

variable "models_bucket" {
  description = "S3 bucket for trained model storage"
  type        = string
  default     = "mlsecops-models-351611731527"
}

variable "mlflow_bucket" {
  description = "S3 bucket for MLflow tracking + artifacts"
  type        = string
  default     = "mlsecops-mlflow-351611731527"
}

variable "instance_type" {
  description = "EC2 instance type for training"
  type        = string
  default     = "c5.xlarge"
}

variable "repo_url" {
  description = "GitHub repo URL for cloning on EC2"
  type        = string
}

variable "mlflow_server_url" {
  description = "MLflow tracking server URL"
  type        = string
  default     = "http://34.193.177.88"
}