# MoleCare ML Infrastructure
# Main Terraform configuration for ML deployment

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state management
  # backend "s3" {
  #   bucket         = "molecare-terraform-state"
  #   key            = "ml/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "molecare-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "molecare"
      ManagedBy = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment (staging/production)"
  type        = string
  default     = "staging"
}

variable "db_password" {
  description = "Database password for Metaflow"
  type        = string
  sensitive   = true
}

# Metaflow Infrastructure
module "metaflow" {
  source = "./modules/metaflow"

  environment = var.environment
  db_password = var.db_password

  tags = {
    Environment = var.environment
  }
}

# ML Lambda (Staging)
module "ml_lambda_staging" {
  source = "./modules/ml-lambda"

  environment        = "staging"
  ecr_repository_url = module.metaflow.ecr_repository_url
  image_tag          = "latest"
  memory_size        = 6144
  timeout            = 60

  tags = {
    Environment = "staging"
  }
}

# ML Lambda (Production) - Only create if environment is production
module "ml_lambda_production" {
  source = "./modules/ml-lambda"
  count  = var.environment == "production" ? 1 : 0

  environment        = "production"
  ecr_repository_url = module.metaflow.ecr_repository_url
  image_tag          = "latest"
  memory_size        = 6144
  timeout            = 60

  tags = {
    Environment = "production"
  }
}

# Outputs
output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = module.metaflow.ecr_repository_url
}

output "metaflow_artifacts_bucket" {
  description = "S3 bucket for Metaflow artifacts"
  value       = module.metaflow.artifacts_bucket
}

output "metaflow_models_bucket" {
  description = "S3 bucket for ML models"
  value       = module.metaflow.models_bucket
}

output "metaflow_db_endpoint" {
  description = "RDS endpoint for Metaflow metadata"
  value       = module.metaflow.db_endpoint
}

output "metaflow_config" {
  description = "Metaflow environment configuration"
  value       = module.metaflow.metaflow_config
}

output "staging_api_endpoint" {
  description = "Staging API endpoint"
  value       = module.ml_lambda_staging.api_endpoint
}

output "staging_predict_url" {
  description = "Staging prediction URL"
  value       = module.ml_lambda_staging.predict_url
}

output "production_api_endpoint" {
  description = "Production API endpoint"
  value       = var.environment == "production" ? module.ml_lambda_production[0].api_endpoint : "Not deployed"
}

output "production_predict_url" {
  description = "Production prediction URL"
  value       = var.environment == "production" ? module.ml_lambda_production[0].predict_url : "Not deployed"
}
