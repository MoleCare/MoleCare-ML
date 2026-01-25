#!/usr/bin/env python3
"""
Canary Deployment Script for MoleCare ML Lambda.

This script manages blue-green deployments using AWS Lambda aliases
and weighted routing for gradual traffic shifting.

Usage:
    # Start canary with 10% traffic
    python canary_deploy.py start --version 5 --weight 0.1

    # Increase traffic to 50%
    python canary_deploy.py update --weight 0.5

    # Complete rollout (100% traffic)
    python canary_deploy.py complete

    # Rollback to previous version
    python canary_deploy.py rollback

    # Check current status
    python canary_deploy.py status
"""
import argparse
import boto3
import json
import time
import wandb
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FUNCTION_NAME = "molecare-ml-prod"
ALIAS_NAME = "live"
ROLLBACK_ALIAS = "rollback"
AWS_REGION = "us-east-1"


class CanaryDeployer:
    """Manages canary deployments for Lambda functions."""

    def __init__(self, function_name: str, region: str = AWS_REGION):
        self.function_name = function_name
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)

    def get_current_alias(self, alias_name: str = ALIAS_NAME) -> Dict[str, Any]:
        """Get current alias configuration."""
        try:
            response = self.lambda_client.get_alias(
                FunctionName=self.function_name,
                Name=alias_name
            )
            return response
        except self.lambda_client.exceptions.ResourceNotFoundException:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        alias = self.get_current_alias()

        if alias is None:
            return {"status": "no_alias", "message": f"Alias '{ALIAS_NAME}' not found"}

        routing = alias.get('RoutingConfig', {}).get('AdditionalVersionWeights', {})
        primary_version = alias['FunctionVersion']

        status = {
            "status": "active",
            "alias": ALIAS_NAME,
            "primary_version": primary_version,
            "primary_weight": 1.0 - sum(routing.values()),
            "canary_versions": routing,
            "is_canary_active": len(routing) > 0
        }

        return status

    def start_canary(self, new_version: str, weight: float = 0.1) -> Dict[str, Any]:
        """
        Start a canary deployment.

        Args:
            new_version: Lambda version number to deploy
            weight: Initial traffic weight (0.0-1.0)

        Returns:
            Deployment status dict
        """
        if weight < 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")

        # Get current alias
        alias = self.get_current_alias()

        if alias is None:
            # Create new alias
            logger.info(f"Creating alias '{ALIAS_NAME}' with version {new_version}")
            self.lambda_client.create_alias(
                FunctionName=self.function_name,
                Name=ALIAS_NAME,
                FunctionVersion=new_version
            )
            return {"status": "created", "version": new_version, "weight": 1.0}

        current_version = alias['FunctionVersion']

        # Save current version as rollback
        self._save_rollback_alias(current_version)

        # Update alias with weighted routing
        logger.info(f"Starting canary: {weight*100}% traffic to version {new_version}")

        self.lambda_client.update_alias(
            FunctionName=self.function_name,
            Name=ALIAS_NAME,
            FunctionVersion=current_version,
            RoutingConfig={
                'AdditionalVersionWeights': {
                    new_version: weight
                }
            }
        )

        # Log to W&B
        self._log_to_wandb({
            "canary/started": True,
            "canary/new_version": new_version,
            "canary/weight": weight,
            "canary/primary_version": current_version
        })

        return {
            "status": "canary_started",
            "primary_version": current_version,
            "canary_version": new_version,
            "canary_weight": weight
        }

    def update_weight(self, weight: float) -> Dict[str, Any]:
        """
        Update canary traffic weight.

        Args:
            weight: New traffic weight (0.0-1.0)

        Returns:
            Updated status dict
        """
        alias = self.get_current_alias()

        if alias is None:
            raise RuntimeError(f"Alias '{ALIAS_NAME}' not found")

        routing = alias.get('RoutingConfig', {}).get('AdditionalVersionWeights', {})

        if not routing:
            raise RuntimeError("No canary deployment active")

        canary_version = list(routing.keys())[0]

        logger.info(f"Updating canary weight to {weight*100}%")

        self.lambda_client.update_alias(
            FunctionName=self.function_name,
            Name=ALIAS_NAME,
            FunctionVersion=alias['FunctionVersion'],
            RoutingConfig={
                'AdditionalVersionWeights': {
                    canary_version: weight
                }
            }
        )

        # Log to W&B
        self._log_to_wandb({
            "canary/weight_updated": weight,
            "canary/version": canary_version
        })

        return {
            "status": "weight_updated",
            "canary_version": canary_version,
            "new_weight": weight
        }

    def complete_rollout(self) -> Dict[str, Any]:
        """
        Complete canary rollout (100% traffic to new version).

        Returns:
            Completion status dict
        """
        alias = self.get_current_alias()

        if alias is None:
            raise RuntimeError(f"Alias '{ALIAS_NAME}' not found")

        routing = alias.get('RoutingConfig', {}).get('AdditionalVersionWeights', {})

        if not routing:
            raise RuntimeError("No canary deployment active")

        canary_version = list(routing.keys())[0]

        logger.info(f"Completing rollout: 100% traffic to version {canary_version}")

        # Save current primary as rollback
        self._save_rollback_alias(alias['FunctionVersion'])

        # Update alias to point to canary version with no routing
        self.lambda_client.update_alias(
            FunctionName=self.function_name,
            Name=ALIAS_NAME,
            FunctionVersion=canary_version,
            RoutingConfig={'AdditionalVersionWeights': {}}
        )

        # Log to W&B
        self._log_to_wandb({
            "canary/completed": True,
            "canary/new_primary_version": canary_version
        })

        return {
            "status": "rollout_complete",
            "new_primary_version": canary_version
        }

    def rollback(self) -> Dict[str, Any]:
        """
        Rollback to previous version.

        Returns:
            Rollback status dict
        """
        # Get rollback version
        rollback_alias = self.get_current_alias(ROLLBACK_ALIAS)

        if rollback_alias is None:
            raise RuntimeError(f"No rollback version available (alias '{ROLLBACK_ALIAS}' not found)")

        rollback_version = rollback_alias['FunctionVersion']

        logger.info(f"Rolling back to version {rollback_version}")

        # Update main alias
        self.lambda_client.update_alias(
            FunctionName=self.function_name,
            Name=ALIAS_NAME,
            FunctionVersion=rollback_version,
            RoutingConfig={'AdditionalVersionWeights': {}}
        )

        # Log to W&B
        self._log_to_wandb({
            "canary/rollback": True,
            "canary/rollback_version": rollback_version
        })

        return {
            "status": "rolled_back",
            "version": rollback_version
        }

    def _save_rollback_alias(self, version: str):
        """Save a version as rollback."""
        try:
            self.lambda_client.update_alias(
                FunctionName=self.function_name,
                Name=ROLLBACK_ALIAS,
                FunctionVersion=version
            )
        except self.lambda_client.exceptions.ResourceNotFoundException:
            self.lambda_client.create_alias(
                FunctionName=self.function_name,
                Name=ROLLBACK_ALIAS,
                FunctionVersion=version
            )
        logger.info(f"Saved version {version} as rollback")

    def _log_to_wandb(self, metrics: Dict[str, Any]):
        """Log metrics to W&B if available."""
        try:
            if wandb.run is None:
                wandb.init(
                    project="molecare-melanoma",
                    name=f"canary-{time.strftime('%Y%m%d-%H%M%S')}",
                    job_type="deployment"
                )
            wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def monitor_canary(self, duration_minutes: int = 60, check_interval_seconds: int = 60) -> Dict[str, Any]:
        """
        Monitor canary deployment metrics.

        Args:
            duration_minutes: How long to monitor
            check_interval_seconds: Check interval

        Returns:
            Monitoring results dict
        """
        logger.info(f"Monitoring canary for {duration_minutes} minutes...")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        errors_detected = False
        latency_issues = False

        while time.time() < end_time:
            # Check CloudWatch metrics
            metrics = self._get_cloudwatch_metrics()

            if metrics.get('error_rate', 0) > 5:
                logger.warning(f"High error rate detected: {metrics['error_rate']}%")
                errors_detected = True

            if metrics.get('latency_p99', 0) > 500:
                logger.warning(f"High latency detected: {metrics['latency_p99']}ms")
                latency_issues = True

            if errors_detected or latency_issues:
                logger.warning("Issues detected, consider rollback")

            time.sleep(check_interval_seconds)

        return {
            "monitored_duration_minutes": duration_minutes,
            "errors_detected": errors_detected,
            "latency_issues": latency_issues,
            "recommendation": "rollback" if (errors_detected or latency_issues) else "proceed"
        }

    def _get_cloudwatch_metrics(self) -> Dict[str, Any]:
        """Get CloudWatch metrics for the Lambda function."""
        # Simplified - would need actual CloudWatch metric queries
        return {
            "error_rate": 0,
            "latency_p99": 0,
            "invocations": 0
        }


def main():
    parser = argparse.ArgumentParser(description="Canary deployment for MoleCare ML")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start canary
    start_parser = subparsers.add_parser("start", help="Start canary deployment")
    start_parser.add_argument("--version", required=True, help="Lambda version to deploy")
    start_parser.add_argument("--weight", type=float, default=0.1, help="Initial traffic weight")

    # Update weight
    update_parser = subparsers.add_parser("update", help="Update canary weight")
    update_parser.add_argument("--weight", type=float, required=True, help="New traffic weight")

    # Complete rollout
    subparsers.add_parser("complete", help="Complete canary rollout")

    # Rollback
    subparsers.add_parser("rollback", help="Rollback to previous version")

    # Status
    subparsers.add_parser("status", help="Get current deployment status")

    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor canary deployment")
    monitor_parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in minutes")

    args = parser.parse_args()

    deployer = CanaryDeployer(FUNCTION_NAME)

    if args.command == "start":
        result = deployer.start_canary(args.version, args.weight)
    elif args.command == "update":
        result = deployer.update_weight(args.weight)
    elif args.command == "complete":
        result = deployer.complete_rollout()
    elif args.command == "rollback":
        result = deployer.rollback()
    elif args.command == "status":
        result = deployer.get_status()
    elif args.command == "monitor":
        result = deployer.monitor_canary(args.duration)
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
