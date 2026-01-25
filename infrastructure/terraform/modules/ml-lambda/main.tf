# MoleCare ML Lambda Infrastructure
# This module creates the Lambda function, API Gateway, and supporting resources

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
  description = "Environment name (staging, production)"
  type        = string
}

variable "ecr_repository_url" {
  description = "ECR repository URL for the ML model container"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "memory_size" {
  description = "Lambda memory size in MB"
  type        = number
  default     = 6144  # 6GB for ML model
}

variable "timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 60
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Locals
locals {
  function_name = "molecare-ml-${var.environment}"
  common_tags = merge(var.tags, {
    Project     = "molecare"
    Component   = "ml"
    Environment = var.environment
  })
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${local.function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for Lambda
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${local.function_name}-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::molecare-ml-models",
          "arn:aws:s3:::molecare-ml-models/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda Function
resource "aws_lambda_function" "ml_function" {
  function_name = local.function_name
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${var.ecr_repository_url}:${var.image_tag}"

  memory_size = var.memory_size
  timeout     = var.timeout

  environment {
    variables = {
      ENVIRONMENT   = var.environment
      MODEL_PATH    = "/var/task/cnn-models/xception/1/"
      LOG_LEVEL     = var.environment == "production" ? "INFO" : "DEBUG"
      WANDB_PROJECT = "molecare-melanoma"
    }
  }

  tags = local.common_tags
}

# Lambda Alias - Live (for production traffic)
resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.ml_function.function_name
  function_version = aws_lambda_function.ml_function.version

  lifecycle {
    ignore_changes = [function_version, routing_config]
  }
}

# Lambda Alias - Rollback (for quick rollback)
resource "aws_lambda_alias" "rollback" {
  name             = "rollback"
  function_name    = aws_lambda_function.ml_function.function_name
  function_version = aws_lambda_function.ml_function.version

  lifecycle {
    ignore_changes = [function_version]
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 30
  tags              = local.common_tags
}

# API Gateway HTTP API
resource "aws_apigatewayv2_api" "ml_api" {
  name          = "molecare-ml-api-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }

  tags = local.common_tags
}

# API Gateway Integration
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.ml_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_alias.live.invoke_arn
  payload_format_version = "2.0"
}

# API Gateway Route - POST /predict
resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.ml_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# API Gateway Route - GET /health
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.ml_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# API Gateway Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.ml_api.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      latency        = "$context.integrationLatency"
    })
  }

  tags = local.common_tags
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/molecare-ml-${var.environment}"
  retention_in_days = 30
  tags              = local.common_tags
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ml_function.function_name
  qualifier     = aws_lambda_alias.live.name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.ml_api.execution_arn}/*/*"
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.function_name}-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Lambda function error rate is too high"

  dimensions = {
    FunctionName = aws_lambda_function.ml_function.function_name
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  alarm_name          = "${local.function_name}-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 60
  extended_statistic  = "p99"
  threshold           = 10000  # 10 seconds
  alarm_description   = "Lambda function P99 latency is too high"

  dimensions = {
    FunctionName = aws_lambda_function.ml_function.function_name
  }

  tags = local.common_tags
}

# Outputs
output "function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.ml_function.function_name
}

output "function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.ml_function.arn
}

output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "predict_url" {
  description = "Full URL for prediction endpoint"
  value       = "${aws_apigatewayv2_stage.default.invoke_url}/predict"
}
