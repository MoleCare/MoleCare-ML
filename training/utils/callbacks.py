"""
Training Callbacks

Provides callbacks for model training including:
- Learning rate scheduling
- Early stopping
- Model checkpointing
- W&B integration
"""

import os
from typing import Callable, List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


def get_lr_schedule(
    initial_lr: float = 1e-4,
    min_lr: float = 1e-6,
    max_lr: float = 2e-4,
    warmup_epochs: int = 5,
    total_epochs: int = 50,
    decay_rate: float = 0.8,
) -> Callable:
    """
    Create a learning rate schedule with warmup and exponential decay.

    Schedule:
    1. Warmup phase: Linear increase from initial_lr to max_lr
    2. Decay phase: Exponential decay from max_lr to min_lr
    """
    def schedule(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Exponential decay
            decay_epochs = epoch - warmup_epochs
            lr = (max_lr - min_lr) * (decay_rate ** decay_epochs) + min_lr

        return max(lr, min_lr)

    return schedule


def get_cosine_schedule(
    initial_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    total_epochs: int = 50,
) -> Callable:
    """
    Create a cosine annealing learning rate schedule with warmup.

    Often works better than exponential decay for fine-tuning.
    """
    import math

    def schedule(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs

        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

    return schedule


def get_training_callbacks(
    model_name: str = 'model',
    output_dir: str = 'outputs',
    patience: int = 10,
    min_delta: float = 0.001,
    monitor: str = 'val_auc',
    mode: str = 'max',
    learning_rate_schedule: Optional[Callable] = None,
    enable_tensorboard: bool = True,
    enable_csv_logger: bool = True,
    wandb_callback: bool = False,
) -> List[tf.keras.callbacks.Callback]:
    """
    Get standard training callbacks.

    Args:
        model_name: Name for saved model files
        output_dir: Directory for outputs
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor for early stopping
        mode: 'max' or 'min' for monitored metric
        learning_rate_schedule: Custom LR schedule function
        enable_tensorboard: Enable TensorBoard logging
        enable_csv_logger: Enable CSV logging
        wandb_callback: Enable W&B callback (requires wandb.init())

    Returns:
        List of Keras callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    callbacks = []

    # Model checkpoint - save best model
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.h5')
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ))

    # Early stopping
    callbacks.append(EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1,
    ))

    # Learning rate scheduler
    if learning_rate_schedule:
        callbacks.append(LearningRateScheduler(
            learning_rate_schedule,
            verbose=1,
        ))
    else:
        # Default: ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        ))

    # TensorBoard
    if enable_tensorboard:
        log_dir = os.path.join(output_dir, 'logs', model_name)
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ))

    # CSV Logger
    if enable_csv_logger:
        csv_path = os.path.join(output_dir, f'{model_name}_training.csv')
        callbacks.append(CSVLogger(
            csv_path,
            separator=',',
            append=False,
        ))

    # W&B callback
    if wandb_callback:
        try:
            import wandb
            from wandb.keras import WandbCallback
            callbacks.append(WandbCallback(
                monitor=monitor,
                mode=mode,
                save_model=False,  # We handle this with ModelCheckpoint
                log_weights=False,
                log_gradients=False,
            ))
        except ImportError:
            print("Warning: wandb not installed, skipping WandbCallback")

    return callbacks


def get_stage1_callbacks(
    model_name: str,
    output_dir: str,
    initial_lr: float = 1e-4,
    epochs: int = 20,
) -> List[tf.keras.callbacks.Callback]:
    """Get callbacks for Stage 1 (head training with frozen base)."""
    lr_schedule = get_lr_schedule(
        initial_lr=initial_lr / 10,
        min_lr=initial_lr / 100,
        max_lr=initial_lr,
        warmup_epochs=3,
        total_epochs=epochs,
    )

    return get_training_callbacks(
        model_name=f'{model_name}_stage1',
        output_dir=output_dir,
        patience=7,
        monitor='val_auc',
        learning_rate_schedule=lr_schedule,
    )


def get_stage2_callbacks(
    model_name: str,
    output_dir: str,
    initial_lr: float = 1e-5,
    epochs: int = 30,
) -> List[tf.keras.callbacks.Callback]:
    """Get callbacks for Stage 2 (fine-tuning with unfrozen layers)."""
    lr_schedule = get_cosine_schedule(
        initial_lr=initial_lr,
        min_lr=initial_lr / 100,
        warmup_epochs=2,
        total_epochs=epochs,
    )

    return get_training_callbacks(
        model_name=f'{model_name}_stage2',
        output_dir=output_dir,
        patience=10,
        monitor='val_auc',
        learning_rate_schedule=lr_schedule,
    )


class MetricsLogger(tf.keras.callbacks.Callback):
    """Custom callback to log detailed metrics after each epoch."""

    def __init__(self, validation_data=None, log_fn=None):
        super().__init__()
        self.validation_data = validation_data
        self.log_fn = log_fn or print

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Log key medical metrics
        metrics_str = f"Epoch {epoch + 1}: "
        metrics_str += f"loss={logs.get('loss', 0):.4f}, "
        metrics_str += f"val_loss={logs.get('val_loss', 0):.4f}, "
        metrics_str += f"auc={logs.get('val_auc', 0):.4f}, "
        metrics_str += f"sensitivity={logs.get('val_sensitivity', logs.get('val_recall', 0)):.4f}"

        self.log_fn(metrics_str)
