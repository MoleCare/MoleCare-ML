"""
Metaflow Deployment Flow for Melanoma Classification Model.

This flow handles model deployment to AWS Lambda.
It supports blue-green deployment via Lambda aliases.

Usage:
    # Deploy to staging
    python deployment_flow.py run --version v1.0.0 --env staging

    # Deploy to production
    python deployment_flow.py run --version v1.0.0 --env production

    # AWS Step Functions
    python deployment_flow.py step-functions create
"""
from metaflow import FlowSpec, step, Parameter, current
import os


class MelanomaDeploymentFlow(FlowSpec):
    """
    Deployment flow for melanoma classification model.

    Steps:
    1. start: Initialize deployment
    2. validate: Validate model artifacts exist
    3. build: Build Docker image for Lambda
    4. deploy: Deploy to Lambda function
    5. verify: Run health checks and smoke tests
    6. promote: Update Lambda alias for traffic routing
    7. end: Finalize deployment
    """

    model_version = Parameter(
        'version',
        help='Model version to deploy',
        required=True
    )

    environment = Parameter(
        'env',
        help='Deployment environment (staging/production)',
        default='staging'
    )

    canary_weight = Parameter(
        'canary_weight',
        help='Initial traffic weight for canary deployment (0.0-1.0)',
        default=0.1
    )

    aws_region = Parameter(
        'region',
        help='AWS region',
        default='us-east-1'
    )

    @step
    def start(self):
        """Initialize the deployment."""
        import wandb

        print(f"Starting deployment for model version: {self.model_version}")
        print(f"Target environment: {self.environment}")

        # Initialize W&B for deployment tracking
        self.wandb_run = wandb.init(
            project="molecare-melanoma",
            name=f"deploy-{self.model_version}-{self.environment}",
            job_type="deployment",
            config={
                "model_version": self.model_version,
                "environment": self.environment,
                "canary_weight": self.canary_weight,
            },
            tags=["deployment", self.environment, self.model_version]
        )

        self.run_id = current.run_id

        # Set function names based on environment
        if self.environment == 'production':
            self.function_name = 'molecare-ml-prod'
            self.api_gateway_stage = 'prod'
        else:
            self.function_name = 'molecare-ml-staging'
            self.api_gateway_stage = 'staging'

        print(f"Lambda function: {self.function_name}")
        self.next(self.validate)

    @step
    def validate(self):
        """Validate model artifacts exist in S3."""
        import boto3
        import wandb

        print(f"Validating model artifacts for version: {self.model_version}")

        # Check S3 for model artifacts
        s3_client = boto3.client('s3', region_name=self.aws_region)
        bucket = 'molecare-ml-models'
        model_key = f"{self.model_version}/model.h5"

        try:
            s3_client.head_object(Bucket=bucket, Key=model_key)
            self.model_exists = True
            print(f"Model found: s3://{bucket}/{model_key}")
        except Exception as e:
            print(f"Warning: Model not found in S3 (will use embedded model): {e}")
            self.model_exists = False

        # Check ECR for Docker image
        ecr_client = boto3.client('ecr', region_name=self.aws_region)
        self.ecr_repository = 'molecare-ml'
        self.image_tag = self.model_version

        try:
            response = ecr_client.describe_images(
                repositoryName=self.ecr_repository,
                imageIds=[{'imageTag': self.image_tag}]
            )
            self.image_exists = True
            self.image_uri = f"{response['imageDetails'][0]['registryId']}.dkr.ecr.{self.aws_region}.amazonaws.com/{self.ecr_repository}:{self.image_tag}"
            print(f"Docker image found: {self.image_uri}")
        except Exception as e:
            print(f"Docker image not found, will need to build: {e}")
            self.image_exists = False
            self.image_uri = None

        wandb.log({
            "validation/model_exists": self.model_exists,
            "validation/image_exists": self.image_exists,
        })

        self.next(self.build)

    @step
    def build(self):
        """Build Docker image for Lambda (if needed)."""
        import wandb

        if self.image_exists:
            print("Docker image already exists, skipping build")
            self.build_skipped = True
        else:
            print("Building Docker image...")
            # In CI/CD, this would trigger the build
            # For now, log that build is needed
            self.build_skipped = False
            print("Note: Docker build should be done in CI/CD pipeline")

        wandb.log({"build/skipped": self.build_skipped})
        self.next(self.deploy)

    @step
    def deploy(self):
        """Deploy to Lambda function."""
        import boto3
        import wandb
        import time

        print(f"Deploying to Lambda function: {self.function_name}")

        lambda_client = boto3.client('lambda', region_name=self.aws_region)

        if self.image_uri:
            try:
                # Update function code with new container image
                response = lambda_client.update_function_code(
                    FunctionName=self.function_name,
                    ImageUri=self.image_uri
                )
                self.function_arn = response['FunctionArn']
                print(f"Function updated: {self.function_arn}")

                # Wait for update to complete
                print("Waiting for function update...")
                waiter = lambda_client.get_waiter('function_updated')
                waiter.wait(FunctionName=self.function_name)
                print("Function update complete")

                # Publish new version
                version_response = lambda_client.publish_version(
                    FunctionName=self.function_name,
                    Description=f"Model version: {self.model_version}"
                )
                self.published_version = version_response['Version']
                print(f"Published version: {self.published_version}")

                self.deploy_success = True

            except Exception as e:
                print(f"Deployment failed: {e}")
                self.deploy_success = False
                self.published_version = None
        else:
            print("No image URI available, deployment skipped")
            self.deploy_success = False
            self.published_version = None

        wandb.log({
            "deploy/success": self.deploy_success,
            "deploy/version": self.published_version,
        })

        self.next(self.verify)

    @step
    def verify(self):
        """Run health checks and smoke tests."""
        import boto3
        import json
        import wandb
        import base64

        print("Running verification tests...")

        self.health_check_passed = False
        self.smoke_test_passed = False

        if not self.deploy_success:
            print("Deployment failed, skipping verification")
            self.next(self.promote)
            return

        lambda_client = boto3.client('lambda', region_name=self.aws_region)

        # Health check
        try:
            response = lambda_client.invoke(
                FunctionName=f"{self.function_name}:{self.published_version}",
                InvocationType='RequestResponse',
                Payload=json.dumps({})
            )
            result = json.loads(response['Payload'].read())
            if result.get('statusCode') == 200:
                self.health_check_passed = True
                print("Health check passed")
            else:
                print(f"Health check failed: {result}")
        except Exception as e:
            print(f"Health check error: {e}")

        # Smoke test with a test image
        try:
            # Create a minimal test image (1x1 pixel)
            test_payload = {
                "predictionid": "smoke-test-" + self.run_id,
                "imagebase64": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAB//9k="
            }

            response = lambda_client.invoke(
                FunctionName=f"{self.function_name}:{self.published_version}",
                InvocationType='RequestResponse',
                Payload=json.dumps(test_payload)
            )
            result = json.loads(response['Payload'].read())
            if result.get('statusCode') == 200:
                self.smoke_test_passed = True
                body = json.loads(result.get('body', '{}'))
                print(f"Smoke test passed: {body}")
            else:
                print(f"Smoke test failed: {result}")
        except Exception as e:
            print(f"Smoke test error: {e}")

        wandb.log({
            "verify/health_check": self.health_check_passed,
            "verify/smoke_test": self.smoke_test_passed,
        })

        self.next(self.promote)

    @step
    def promote(self):
        """Update Lambda alias for traffic routing."""
        import boto3
        import wandb

        print(f"Promoting deployment to alias...")

        if not self.deploy_success or not self.health_check_passed:
            print("Deployment or verification failed, skipping promotion")
            self.promotion_success = False
            self.next(self.end)
            return

        lambda_client = boto3.client('lambda', region_name=self.aws_region)

        # Determine alias based on environment
        if self.environment == 'production':
            alias_name = 'live'
        else:
            alias_name = 'staging'

        try:
            # Check if alias exists
            try:
                lambda_client.get_alias(
                    FunctionName=self.function_name,
                    Name=alias_name
                )
                alias_exists = True
            except:
                alias_exists = False

            if self.environment == 'production' and self.canary_weight > 0:
                # Canary deployment for production
                if alias_exists:
                    # Get current version
                    current_alias = lambda_client.get_alias(
                        FunctionName=self.function_name,
                        Name=alias_name
                    )
                    current_version = current_alias['FunctionVersion']

                    # Update with weighted routing
                    lambda_client.update_alias(
                        FunctionName=self.function_name,
                        Name=alias_name,
                        FunctionVersion=current_version,
                        RoutingConfig={
                            'AdditionalVersionWeights': {
                                self.published_version: self.canary_weight
                            }
                        }
                    )
                    print(f"Canary deployment: {self.canary_weight*100}% traffic to version {self.published_version}")
                else:
                    # Create new alias
                    lambda_client.create_alias(
                        FunctionName=self.function_name,
                        Name=alias_name,
                        FunctionVersion=self.published_version
                    )
                    print(f"Created alias '{alias_name}' pointing to version {self.published_version}")
            else:
                # Direct deployment (staging or full production rollout)
                if alias_exists:
                    lambda_client.update_alias(
                        FunctionName=self.function_name,
                        Name=alias_name,
                        FunctionVersion=self.published_version,
                        RoutingConfig={'AdditionalVersionWeights': {}}
                    )
                else:
                    lambda_client.create_alias(
                        FunctionName=self.function_name,
                        Name=alias_name,
                        FunctionVersion=self.published_version
                    )
                print(f"Alias '{alias_name}' now points to version {self.published_version}")

            self.promotion_success = True

        except Exception as e:
            print(f"Promotion failed: {e}")
            self.promotion_success = False

        wandb.log({
            "promote/success": self.promotion_success,
            "promote/alias": alias_name,
            "promote/version": self.published_version,
        })

        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment."""
        import wandb

        success = self.deploy_success and self.health_check_passed and self.promotion_success

        print(f"\nDeployment Summary:")
        print(f"  Model Version: {self.model_version}")
        print(f"  Environment: {self.environment}")
        print(f"  Deploy Success: {self.deploy_success}")
        print(f"  Health Check: {self.health_check_passed}")
        print(f"  Smoke Test: {self.smoke_test_passed}")
        print(f"  Promotion: {self.promotion_success}")
        print(f"  Overall: {'SUCCESS' if success else 'FAILED'}")

        # Log final summary
        wandb.run.summary["deployment_success"] = success
        wandb.run.summary["model_version"] = self.model_version
        wandb.run.summary["environment"] = self.environment

        wandb.finish()


if __name__ == '__main__':
    MelanomaDeploymentFlow()
