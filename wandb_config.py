"""
Weights & Biases Configuration for MoleCare ML.

This module provides centralized W&B configuration and utility functions
for experiment tracking, model monitoring, and inference logging.

Setup:
    1. Create W&B account at wandb.ai
    2. Set WANDB_API_KEY environment variable
    3. Import and use these functions in your code

Usage:
    from wandb_config import init_wandb, log_inference, log_training_metrics

    # For training
    init_wandb("training-run-001", job_type="training")
    log_training_metrics(epoch=1, loss=0.5, accuracy=0.85)

    # For inference monitoring
    init_wandb("inference-monitoring", job_type="inference")
    log_inference(prediction="melanoma", confidence=0.92, latency_ms=150)
"""
import wandb
import os
import time
from typing import Dict, Any, Optional, List
from functools import wraps


# Project configuration
WANDB_PROJECT = "molecare-melanoma"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)  # Optional: your W&B team/organization

# Default model configuration
DEFAULT_CONFIG = {
    "model": "Xception",
    "input_size": 299,
    "batch_size": 32,
    "learning_rate": 0.001,
}


def init_wandb(
    run_name: str,
    job_type: str = "training",
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    resume: bool = False
) -> wandb.run:
    """
    Initialize a W&B run with MoleCare defaults.

    Args:
        run_name: Name for this run
        job_type: Type of job (training, inference, evaluation, deployment)
        config: Configuration dict to log
        tags: List of tags for filtering runs
        notes: Optional notes about this run
        resume: Whether to resume a previous run

    Returns:
        wandb.run object
    """
    merged_config = {**DEFAULT_CONFIG, **(config or {})}

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        job_type=job_type,
        config=merged_config,
        tags=tags or [],
        notes=notes,
        resume="allow" if resume else None
    )


def log_inference(
    prediction: str,
    confidence: float,
    latency_ms: float,
    prediction_id: Optional[str] = None,
    image_size_kb: Optional[float] = None,
    extra_metrics: Optional[Dict[str, Any]] = None
):
    """
    Log inference metrics to W&B.

    Args:
        prediction: Prediction result ("melanoma" or "not_melanoma")
        confidence: Prediction confidence (0-1)
        latency_ms: Inference latency in milliseconds
        prediction_id: Optional prediction ID for tracking
        image_size_kb: Optional image size in KB
        extra_metrics: Optional additional metrics to log
    """
    metrics = {
        "inference/prediction": prediction,
        "inference/confidence": confidence,
        "inference/latency_ms": latency_ms,
        "inference/timestamp": time.time(),
    }

    if prediction_id:
        metrics["inference/prediction_id"] = prediction_id

    if image_size_kb:
        metrics["inference/image_size_kb"] = image_size_kb

    if extra_metrics:
        for key, value in extra_metrics.items():
            metrics[f"inference/{key}"] = value

    wandb.log(metrics)


def log_training_metrics(
    epoch: int,
    loss: float,
    accuracy: float,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    learning_rate: Optional[float] = None,
    extra_metrics: Optional[Dict[str, Any]] = None
):
    """
    Log training metrics to W&B.

    Args:
        epoch: Current epoch number
        loss: Training loss
        accuracy: Training accuracy
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        learning_rate: Current learning rate
        extra_metrics: Optional additional metrics
    """
    metrics = {
        "train/epoch": epoch,
        "train/loss": loss,
        "train/accuracy": accuracy,
    }

    if val_loss is not None:
        metrics["val/loss"] = val_loss

    if val_accuracy is not None:
        metrics["val/accuracy"] = val_accuracy

    if learning_rate is not None:
        metrics["train/learning_rate"] = learning_rate

    if extra_metrics:
        for key, value in extra_metrics.items():
            metrics[f"train/{key}"] = value

    wandb.log(metrics)


def log_evaluation_metrics(
    accuracy: float,
    auc: float,
    precision: float,
    recall: float,
    f1_score: float,
    confusion_matrix: Optional[List[List[int]]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None
):
    """
    Log evaluation metrics to W&B.

    Args:
        accuracy: Model accuracy
        auc: Area under ROC curve
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        confusion_matrix: Optional confusion matrix [[TN, FP], [FN, TP]]
        extra_metrics: Optional additional metrics
    """
    metrics = {
        "eval/accuracy": accuracy,
        "eval/auc": auc,
        "eval/precision": precision,
        "eval/recall": recall,
        "eval/f1_score": f1_score,
    }

    if extra_metrics:
        for key, value in extra_metrics.items():
            metrics[f"eval/{key}"] = value

    wandb.log(metrics)

    # Log confusion matrix as a plot
    if confusion_matrix:
        wandb.log({
            "eval/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=[0] * confusion_matrix[0][0] + [0] * confusion_matrix[0][1] +
                       [1] * confusion_matrix[1][0] + [1] * confusion_matrix[1][1],
                preds=[0] * confusion_matrix[0][0] + [1] * confusion_matrix[0][1] +
                       [0] * confusion_matrix[1][0] + [1] * confusion_matrix[1][1],
                class_names=["NotMelanoma", "Melanoma"]
            )
        })


def log_model_artifact(
    model_path: str,
    model_name: str,
    model_version: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a model as a W&B artifact.

    Args:
        model_path: Path to the saved model
        model_name: Name for the artifact
        model_version: Version string (e.g., "v1.0.0")
        metadata: Optional metadata dict
    """
    artifact = wandb.Artifact(
        name=f"{model_name}-{model_version}",
        type="model",
        metadata=metadata or {}
    )

    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)


def track_inference(func):
    """
    Decorator to automatically track inference metrics.

    Usage:
        @track_inference
        def predict(image_base64):
            # prediction logic
            return {"prediction": "melanoma", "confidence": 0.92}
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        latency_ms = (time.time() - start_time) * 1000

        if isinstance(result, dict):
            log_inference(
                prediction=result.get("prediction", "unknown"),
                confidence=result.get("confidence", 0),
                latency_ms=latency_ms,
                prediction_id=result.get("predictionid")
            )

        return result

    return wrapper


def finish():
    """Finish the current W&B run."""
    wandb.finish()


# Alert thresholds for monitoring
class AlertThresholds:
    """Configurable alert thresholds for model monitoring."""

    LATENCY_P99_MS = 500  # Alert if P99 latency exceeds 500ms
    ERROR_RATE_PERCENT = 5  # Alert if error rate exceeds 5%
    CONFIDENCE_LOW = 0.3  # Alert if many predictions have low confidence
    DATA_DRIFT_KL = 0.1  # Alert if KL divergence exceeds 0.1


def check_alerts(metrics: Dict[str, Any]) -> List[str]:
    """
    Check metrics against alert thresholds.

    Args:
        metrics: Dict of current metrics

    Returns:
        List of alert messages (empty if no alerts)
    """
    alerts = []

    if metrics.get("latency_p99_ms", 0) > AlertThresholds.LATENCY_P99_MS:
        alerts.append(f"High latency: P99 = {metrics['latency_p99_ms']}ms")

    if metrics.get("error_rate", 0) > AlertThresholds.ERROR_RATE_PERCENT:
        alerts.append(f"High error rate: {metrics['error_rate']}%")

    if metrics.get("low_confidence_rate", 0) > 0.1:
        alerts.append(f"Many low confidence predictions: {metrics['low_confidence_rate']*100}%")

    return alerts
