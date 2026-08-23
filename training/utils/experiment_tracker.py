"""
Experiment Tracker

Unified interface for experiment tracking with W&B integration.
Tracks metrics, hyperparameters, and model artifacts.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class ExperimentTracker:
    """
    Unified experiment tracking for ML training.

    Supports:
    - Local JSON logging
    - Weights & Biases integration
    - Metric comparison across experiments
    """

    def __init__(
        self,
        experiment_name: str,
        project_name: str = 'molecare-melanoma',
        output_dir: str = 'outputs',
        use_wandb: bool = True,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.config = config or {}
        self.tags = tags or []

        self.metrics_history = []
        self.artifacts = []
        self.wandb_run = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize W&B if enabled
        if use_wandb:
            self._init_wandb(wandb_entity)

        # Log start time
        self.start_time = datetime.now()
        self.log_event('experiment_started', {'timestamp': self.start_time.isoformat()})

    def _init_wandb(self, entity: Optional[str] = None):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                entity=entity,
                config=self.config,
                tags=self.tags,
                reinit=True,
            )
            print(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")
            self.use_wandb = False

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config.update(config)

        if self.use_wandb:
            import wandb
            wandb.config.update(config)

        # Save to local file
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current step/epoch."""
        metrics_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log metrics for a training epoch."""
        metrics = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        self.log_metrics(metrics, step=epoch)

    def log_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """Log a custom event."""
        event = {
            'event': event_name,
            'timestamp': datetime.now().isoformat(),
            'data': data or {},
        }

        events_path = os.path.join(self.experiment_dir, 'events.jsonl')
        with open(events_path, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def log_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a model artifact."""
        artifact_info = {
            'path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        }
        self.artifacts.append(artifact_info)

        if self.use_wandb:
            import wandb
            artifact = wandb.Artifact(
                name=f'{self.experiment_name}_model',
                type='model',
                metadata={**metrics, **(metadata or {})},
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
    ):
        """Log confusion matrix visualization."""
        if self.use_wandb:
            import wandb
            class_names = class_names or ['Melanoma', 'NotMelanoma']
            wandb.log({
                'confusion_matrix': wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            })

    def log_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Log ROC curve."""
        if self.use_wandb:
            import wandb
            # Invert for Melanoma as positive class
            wandb.log({
                'roc_curve': wandb.plot.roc_curve(
                    1 - y_true,
                    np.column_stack([y_prob, 1 - y_prob]),
                    labels=['NotMelanoma', 'Melanoma'],
                )
            })

    def log_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Log Precision-Recall curve."""
        if self.use_wandb:
            import wandb
            wandb.log({
                'pr_curve': wandb.plot.pr_curve(
                    1 - y_true,
                    np.column_stack([y_prob, 1 - y_prob]),
                    labels=['NotMelanoma', 'Melanoma'],
                )
            })

    def log_image_samples(
        self,
        images: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        caption_prefix: str = 'Sample',
    ):
        """Log sample images with predictions."""
        if self.use_wandb:
            import wandb
            wandb_images = []
            for i, (img, pred, label) in enumerate(zip(images, predictions, labels)):
                pred_class = 'Melanoma' if pred < 0.5 else 'NotMelanoma'
                true_class = 'Melanoma' if label == 0 else 'NotMelanoma'
                caption = f"{caption_prefix} {i}: Pred={pred_class}, True={true_class}"
                wandb_images.append(wandb.Image(img, caption=caption))
            wandb.log({'sample_predictions': wandb_images})

    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from training history."""
        if not self.metrics_history:
            return {}

        best = {
            'best_val_auc': max((m.get('val_auc', 0) for m in self.metrics_history), default=0),
            'best_val_accuracy': max((m.get('val_accuracy', 0) for m in self.metrics_history), default=0),
            'best_val_sensitivity': max((m.get('val_sensitivity', m.get('val_recall', 0)) for m in self.metrics_history), default=0),
            'min_val_loss': min((m.get('val_loss', float('inf')) for m in self.metrics_history), default=0),
        }
        return best

    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """Finish experiment and save summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'config': self.config,
            'best_metrics': self.get_best_metrics(),
            'final_metrics': final_metrics or {},
            'artifacts': self.artifacts,
        }

        # Save summary
        summary_path = os.path.join(self.experiment_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save metrics history
        history_path = os.path.join(self.experiment_dir, 'metrics_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

        # Log final summary to W&B
        if self.use_wandb:
            import wandb
            wandb.run.summary.update({
                'duration_seconds': duration,
                **self.get_best_metrics(),
                **(final_metrics or {}),
            })
            wandb.finish()

        self.log_event('experiment_finished', {'duration_seconds': duration})
        print(f"Experiment finished. Summary saved to {summary_path}")

        return summary


def compare_experiments(
    experiment_dirs: List[str],
    metrics_to_compare: List[str] = None,
) -> Dict[str, Any]:
    """
    Compare multiple experiments.

    Args:
        experiment_dirs: List of experiment output directories
        metrics_to_compare: Metrics to include in comparison

    Returns:
        Comparison dictionary with ranked experiments
    """
    metrics_to_compare = metrics_to_compare or [
        'best_val_auc',
        'best_val_accuracy',
        'best_val_sensitivity',
        'min_val_loss',
    ]

    experiments = []

    for exp_dir in experiment_dirs:
        summary_path = os.path.join(exp_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                experiments.append(summary)

    if not experiments:
        return {'error': 'No experiments found'}

    # Rank by AUC
    experiments_sorted = sorted(
        experiments,
        key=lambda x: x.get('best_metrics', {}).get('best_val_auc', 0),
        reverse=True,
    )

    comparison = {
        'experiments': [{
            'name': exp['experiment_name'],
            'model': exp.get('config', {}).get('model_name', 'Unknown'),
            **{m: exp.get('best_metrics', {}).get(m, 0) for m in metrics_to_compare},
        } for exp in experiments_sorted],
        'best_experiment': experiments_sorted[0]['experiment_name'] if experiments_sorted else None,
    }

    return comparison
