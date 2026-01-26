# MoleCare ML Infrastructure Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment (staging/production)"
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be 'staging' or 'production'."
  }
}

variable "db_password" {
  description = "Database password for Metaflow RDS"
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class for Metaflow metadata"
  type        = string
  default     = "db.t3.micro"
}

variable "lambda_memory_size" {
  description = "Lambda memory size in MB (model requires ~4-6GB)"
  type        = number
  default     = 6144

  validation {
    condition     = var.lambda_memory_size >= 1024 && var.lambda_memory_size <= 10240
    error_message = "Lambda memory must be between 1024 and 10240 MB."
  }
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 60

  validation {
    condition     = var.lambda_timeout >= 1 && var.lambda_timeout <= 900
    error_message = "Lambda timeout must be between 1 and 900 seconds."
  }
}

variable "enable_production" {
  description = "Enable production environment deployment"
  type        = bool
  default     = false
}

variable "domain_name" {
  description = "Custom domain name for API Gateway (optional)"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for custom domain (required if domain_name is set)"
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Email address for CloudWatch alerts"
  type        = string
  default     = ""
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

variable "wandb_api_key" {
  description = "Weights & Biases API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
