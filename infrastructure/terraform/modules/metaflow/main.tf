# MoleCare Metaflow Infrastructure
# This module creates the S3 bucket, RDS database, and IAM roles for Metaflow

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"  # ~$15/month
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "vpc_id" {
  description = "VPC ID for RDS"
  type        = string
  default     = null  # Will use default VPC if not specified
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Locals
locals {
  prefix = "molecare-ml"
  common_tags = merge(var.tags, {
    Project     = "molecare"
    Component   = "metaflow"
    Environment = var.environment
  })
}

# Data source for default VPC
data "aws_vpc" "default" {
  default = var.vpc_id == null ? true : false
  id      = var.vpc_id
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# S3 Bucket for Metaflow artifacts
resource "aws_s3_bucket" "metaflow_artifacts" {
  bucket = "${local.prefix}-artifacts-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "metaflow_artifacts" {
  bucket = aws_s3_bucket.metaflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "metaflow_artifacts" {
  bucket = aws_s3_bucket.metaflow_artifacts.id

  rule {
    id     = "cleanup-old-artifacts"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "metaflow_artifacts" {
  bucket = aws_s3_bucket.metaflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket for ML models
resource "aws_s3_bucket" "ml_models" {
  bucket = "${local.prefix}-models-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Security Group for RDS
resource "aws_security_group" "metaflow_db" {
  name        = "${local.prefix}-metaflow-db-sg"
  description = "Security group for Metaflow RDS"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# DB Subnet Group
resource "aws_db_subnet_group" "metaflow" {
  name       = "${local.prefix}-metaflow-subnet-group"
  subnet_ids = data.aws_subnets.default.ids
  tags       = local.common_tags
}

# RDS PostgreSQL for Metaflow metadata
resource "aws_db_instance" "metaflow_db" {
  identifier = "${local.prefix}-metaflow-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "metaflow"
  username = "metaflow"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.metaflow_db.id]
  db_subnet_group_name   = aws_db_subnet_group.metaflow.name

  publicly_accessible = false
  skip_final_snapshot = true

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  tags = local.common_tags
}

# IAM Role for Metaflow
resource "aws_iam_role" "metaflow_role" {
  name = "${local.prefix}-metaflow-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "states.amazonaws.com",
            "batch.amazonaws.com",
            "lambda.amazonaws.com"
          ]
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Condition = {
          StringEquals = {
            "aws:PrincipalTag/metaflow" = "true"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for Metaflow
resource "aws_iam_role_policy" "metaflow_policy" {
  name = "${local.prefix}-metaflow-policy"
  role = aws_iam_role.metaflow_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.metaflow_artifacts.arn,
          "${aws_s3_bucket.metaflow_artifacts.arn}/*",
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution",
          "states:DescribeExecution",
          "states:StopExecution",
          "states:ListExecutions"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateAlias",
          "lambda:GetFunction",
          "lambda:PublishVersion"
        ]
        Resource = "arn:aws:lambda:*:*:function:molecare-ml-*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = "*"
      }
    ]
  })
}

# ECR Repository for ML model images
resource "aws_ecr_repository" "ml_repository" {
  name                 = "molecare-ml"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "ml_repository" {
  repository = aws_ecr_repository.ml_repository.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# Step Functions State Machine for Metaflow
resource "aws_sfn_state_machine" "metaflow" {
  name     = "${local.prefix}-metaflow-workflow"
  role_arn = aws_iam_role.metaflow_role.arn

  definition = jsonencode({
    Comment = "Metaflow workflow orchestration"
    StartAt = "Initialize"
    States = {
      Initialize = {
        Type = "Pass"
        Next = "ProcessFlow"
      }
      ProcessFlow = {
        Type = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = "molecare-ml-metaflow-runner"
          "Payload.$" = "$"
        }
        End = true
      }
    }
  })

  tags = local.common_tags
}

# Outputs
output "artifacts_bucket" {
  description = "S3 bucket for Metaflow artifacts"
  value       = aws_s3_bucket.metaflow_artifacts.bucket
}

output "models_bucket" {
  description = "S3 bucket for ML models"
  value       = aws_s3_bucket.ml_models.bucket
}

output "db_endpoint" {
  description = "RDS endpoint for Metaflow metadata"
  value       = aws_db_instance.metaflow_db.endpoint
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.metaflow_db.db_name
}

output "metaflow_role_arn" {
  description = "IAM role ARN for Metaflow"
  value       = aws_iam_role.metaflow_role.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.ml_repository.repository_url
}

output "state_machine_arn" {
  description = "Step Functions state machine ARN"
  value       = aws_sfn_state_machine.metaflow.arn
}

# Metaflow configuration output
output "metaflow_config" {
  description = "Metaflow configuration for AWS"
  value = {
    METAFLOW_DATASTORE_SYSROOT_S3 = "s3://${aws_s3_bucket.metaflow_artifacts.bucket}/metaflow"
    METAFLOW_DATATOOLS_S3ROOT     = "s3://${aws_s3_bucket.metaflow_artifacts.bucket}/data"
    METAFLOW_DEFAULT_DATASTORE    = "s3"
    METAFLOW_SFN_STATE_MACHINE    = aws_sfn_state_machine.metaflow.arn
    METAFLOW_SFN_ROLE_ARN         = aws_iam_role.metaflow_role.arn
  }
}
