"""
Training Configuration

Centralized configuration for all training experiments.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_name: str = "EfficientNetB4"
    model_version: str = "v1.0.0"
    input_shape: tuple = (380, 380)  # EfficientNetB4 optimal
    num_classes: int = 1  # Binary classification

    # Training hyperparameters
    batch_size: int = 16
    epochs_stage1: int = 20  # Head training (frozen base)
    epochs_stage2: int = 30  # Fine-tuning (unfrozen top layers)

    # Learning rate schedule
    initial_lr: float = 1e-4
    min_lr: float = 1e-6
    max_lr: float = 2e-4
    warmup_epochs: int = 5

    # Regularization
    dropout_rate: float = 0.3
    l2_regularization: float = 0.001
    label_smoothing: float = 0.1

    # Data augmentation
    enable_augmentation: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.1
    zoom_range: float = 0.15
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_range: tuple = (0.8, 1.2)

    # Data split
    validation_split: float = 0.15
    test_split: float = 0.15

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Paths
    data_dir: str = "data/"
    output_dir: str = "outputs/"
    model_dir: str = "models/"

    # Experiment tracking
    wandb_project: str = "molecare-melanoma"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None

    # Hardware
    mixed_precision: bool = False
    use_tpu: bool = False

    # Deployment thresholds
    deploy_threshold_auc: float = 0.90
    deploy_threshold_sensitivity: float = 0.85

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_{self.model_version}"

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'input_shape': self.input_shape,
            'batch_size': self.batch_size,
            'epochs_stage1': self.epochs_stage1,
            'epochs_stage2': self.epochs_stage2,
            'initial_lr': self.initial_lr,
            'dropout_rate': self.dropout_rate,
            'enable_augmentation': self.enable_augmentation,
            'deploy_threshold_auc': self.deploy_threshold_auc,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items()
                     if k in cls.__dataclass_fields__})


# Model-specific input shapes (optimal for each architecture)
MODEL_INPUT_SHAPES = {
    'EfficientNetB4': (380, 380),
    'EfficientNetB3': (300, 300),
    'EfficientNetV2S': (384, 384),
    'EfficientNetV2M': (480, 480),
    'Xception': (299, 299),
    'InceptionV3': (299, 299),
    'InceptionResNetV2': (299, 299),
    'DenseNet201': (224, 224),
    'DenseNet169': (224, 224),
    'ResNet50V2': (224, 224),
    'VGG16': (224, 224),
}


def get_config_for_model(model_name: str, **overrides) -> TrainingConfig:
    """Get optimized config for a specific model."""
    input_shape = MODEL_INPUT_SHAPES.get(model_name, (299, 299))

    # Model-specific adjustments
    config_kwargs = {
        'model_name': model_name,
        'input_shape': input_shape,
    }

    # EfficientNet models need larger batch size
    if 'EfficientNet' in model_name:
        config_kwargs['batch_size'] = 16
        config_kwargs['dropout_rate'] = 0.4

    # VGG needs smaller learning rate
    if model_name == 'VGG16':
        config_kwargs['initial_lr'] = 5e-5
        config_kwargs['max_lr'] = 1e-4

    # Apply overrides
    config_kwargs.update(overrides)

    return TrainingConfig(**config_kwargs)
