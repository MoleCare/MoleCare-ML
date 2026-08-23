"""
Model Factory

Creates and configures models for melanoma classification.
Addresses key issues from original experiments:
- Fixes loss function (removed from_logits=True with sigmoid)
- Supports two-stage training (head + fine-tuning)
- Includes all tested architectures
"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import (
    VGG16,
    DenseNet169,
    DenseNet201,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetV2M,
    EfficientNetV2S,
    InceptionResNetV2,
    InceptionV3,
    ResNet50V2,
    Xception,
)

# Supported models with their default input shapes
SUPPORTED_MODELS = {
    'Xception': {'class': Xception, 'input_shape': (299, 299), 'preprocess': 'xception'},
    'InceptionV3': {'class': InceptionV3, 'input_shape': (299, 299), 'preprocess': 'inception_v3'},
    'InceptionResNetV2': {'class': InceptionResNetV2, 'input_shape': (299, 299), 'preprocess': 'inception_resnet_v2'},
    'DenseNet201': {'class': DenseNet201, 'input_shape': (224, 224), 'preprocess': 'densenet'},
    'DenseNet169': {'class': DenseNet169, 'input_shape': (224, 224), 'preprocess': 'densenet'},
    'VGG16': {'class': VGG16, 'input_shape': (224, 224), 'preprocess': 'vgg16'},
    'ResNet50V2': {'class': ResNet50V2, 'input_shape': (224, 224), 'preprocess': 'resnet_v2'},
    'EfficientNetB3': {'class': EfficientNetB3, 'input_shape': (300, 300), 'preprocess': 'efficientnet'},
    'EfficientNetB4': {'class': EfficientNetB4, 'input_shape': (380, 380), 'preprocess': 'efficientnet'},
    'EfficientNetV2S': {'class': EfficientNetV2S, 'input_shape': (384, 384), 'preprocess': 'efficientnet_v2'},
    'EfficientNetV2M': {'class': EfficientNetV2M, 'input_shape': (480, 480), 'preprocess': 'efficientnet_v2'},
}


class ModelFactory:
    """Factory for creating melanoma classification models."""

    def __init__(
        self,
        model_name: str = 'EfficientNetB4',
        input_shape: Optional[Tuple[int, int]] = None,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        label_smoothing: float = 0.1,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. "
                           f"Choose from: {list(SUPPORTED_MODELS.keys())}")

        self.model_name = model_name
        self.model_config = SUPPORTED_MODELS[model_name]
        self.input_shape = input_shape or self.model_config['input_shape']
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.label_smoothing = label_smoothing

    def create_base_model(self) -> Model:
        """Create the base model with ImageNet weights."""
        ModelClass = self.model_config['class']

        base_model = ModelClass(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_shape, 3),
        )

        # Freeze base model for initial training
        base_model.trainable = False

        return base_model

    def create_model(self, compile_model: bool = True) -> Model:
        """
        Create the full classification model.

        Architecture:
        - Pre-trained base (frozen initially)
        - Global Average Pooling
        - Dropout for regularization
        - Dense layer with L2 regularization
        - Dropout
        - Output layer with sigmoid

        Note: Using sigmoid activation WITHOUT from_logits=True in loss.
        The original experiments had this bug (sigmoid + from_logits=True).
        """
        base_model = self.create_base_model()

        # Build classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            layers.Dropout(self.dropout_rate),
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            layers.Dropout(self.dropout_rate / 2),
            # Sigmoid output for binary classification
            layers.Dense(1, activation='sigmoid')
        ], name=f'{self.model_name}_melanoma')

        if compile_model:
            model = self.compile_model(model)

        return model

    def compile_model(
        self,
        model: Model,
        learning_rate: float = 1e-4,
        stage: str = 'head',
    ) -> Model:
        """
        Compile the model with appropriate optimizer and loss.

        Important: Using BinaryCrossentropy WITHOUT from_logits=True
        because we have sigmoid activation in the output layer.
        """
        # Use different learning rates for different stages
        if stage == 'fine_tuning':
            learning_rate = learning_rate / 10

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # CORRECT loss configuration:
        # - sigmoid activation in output layer
        # - from_logits=False (default) in loss
        # - label_smoothing for regularization
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,  # Important: sigmoid output, not logits
            label_smoothing=self.label_smoothing,
        )

        # Comprehensive metrics for medical classification
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),  # Sensitivity
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),  # Precision-Recall AUC
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model

    def unfreeze_for_fine_tuning(
        self,
        model: Model,
        num_layers_to_unfreeze: int = 30,
        learning_rate: float = 1e-5,
    ) -> Model:
        """
        Unfreeze top layers of base model for fine-tuning.

        This is critical for achieving high accuracy - the original
        experiments kept the base model completely frozen.
        """
        # Get the base model (first layer in Sequential)
        base_model = model.layers[0]

        # Unfreeze the base model
        base_model.trainable = True

        # Freeze all layers except the last N
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        # Count trainable parameters
        trainable_params = sum([
            tf.keras.backend.count_params(w)
            for w in model.trainable_weights
        ])
        print(f"Unfrozen {num_layers_to_unfreeze} layers, "
              f"trainable params: {trainable_params:,}")

        # Recompile with lower learning rate
        model = self.compile_model(model, learning_rate, stage='fine_tuning')

        return model

    def get_model_summary(self, model: Model) -> Dict[str, Any]:
        """Get model summary information."""
        trainable = sum([tf.keras.backend.count_params(w)
                        for w in model.trainable_weights])
        non_trainable = sum([tf.keras.backend.count_params(w)
                            for w in model.non_trainable_weights])

        return {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'total_params': trainable + non_trainable,
            'trainable_params': trainable,
            'non_trainable_params': non_trainable,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_reg,
        }


def create_model(
    model_name: str = 'EfficientNetB4',
    input_shape: Optional[Tuple[int, int]] = None,
    compile_model: bool = True,
    **kwargs
) -> Tuple[Model, ModelFactory]:
    """
    Convenience function to create a model.

    Returns:
        Tuple of (model, factory) for further operations like fine-tuning.
    """
    factory = ModelFactory(model_name, input_shape, **kwargs)
    model = factory.create_model(compile_model)
    return model, factory
