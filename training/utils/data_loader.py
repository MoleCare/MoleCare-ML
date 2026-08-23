"""
Data Loading and Augmentation

Handles data loading, preprocessing, and augmentation for melanoma classification.
Addresses key issues from original experiments:
- Adds proper data augmentation (was missing)
- Uses tf.data for better performance
- Supports S3 and local data sources
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    """Handles data loading and augmentation for melanoma classification."""

    def __init__(
        self,
        data_dir: str,
        input_shape: Tuple[int, int] = (299, 299),
        batch_size: int = 16,
        enable_augmentation: bool = True,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.enable_augmentation = enable_augmentation
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed

        # Class names
        self.class_names = ['Melanoma', 'NotMelanoma']
        self.class_indices = {'Melanoma': 0, 'NotMelanoma': 1}

    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get data augmentation configuration.

        These augmentations are specifically designed for skin lesion images:
        - Rotation: Lesions can appear at any angle
        - Flips: Vertical/horizontal symmetry doesn't change diagnosis
        - Zoom: Captures lesions at different scales
        - Brightness: Accounts for different lighting conditions
        - Shear: Minor geometric distortions
        """
        if not self.enable_augmentation:
            return {'rescale': 1./255.}

        return {
            'rescale': 1./255.,
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.15,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': (0.8, 1.2),
            'fill_mode': 'reflect',
        }

    def create_generators(
        self,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
    ) -> Tuple:
        """
        Create data generators for training, validation, and testing.

        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        train_dir = train_dir or os.path.join(self.data_dir, 'training')
        val_dir = val_dir or os.path.join(self.data_dir, 'validation')
        test_dir = test_dir or os.path.join(self.data_dir, 'testing')

        # Training generator WITH augmentation
        train_datagen = ImageDataGenerator(**self.get_augmentation_config())

        # Validation/Test generators WITHOUT augmentation (only rescale)
        eval_datagen = ImageDataGenerator(rescale=1./255.)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=self.seed,
        )

        val_generator = eval_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
        )

        test_generator = eval_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
        )

        return train_generator, val_generator, test_generator

    def create_tf_datasets(
        self,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create tf.data.Dataset pipelines for better performance.

        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        train_dir = train_dir or os.path.join(self.data_dir, 'training')
        val_dir = val_dir or os.path.join(self.data_dir, 'validation')
        test_dir = test_dir or os.path.join(self.data_dir, 'testing')

        AUTOTUNE = tf.data.AUTOTUNE

        # Load datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            label_mode='binary',
            shuffle=True,
            seed=self.seed,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            label_mode='binary',
            shuffle=False,
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            label_mode='binary',
            shuffle=False,
        )

        # Normalization layer
        normalization = tf.keras.layers.Rescaling(1./255)

        # Augmentation layer (only for training)
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ]) if self.enable_augmentation else None

        def prepare_train(image, label):
            image = normalization(image)
            if augmentation:
                image = augmentation(image, training=True)
            return image, label

        def prepare_eval(image, label):
            image = normalization(image)
            return image, label

        # Apply preprocessing
        train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)

        # Optimize performance
        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds, test_ds

    def calculate_class_weights(self, train_dir: Optional[str] = None) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.

        Returns:
            Dictionary mapping class index to weight
        """
        train_dir = train_dir or os.path.join(self.data_dir, 'training')

        melanoma_count = len(os.listdir(os.path.join(train_dir, 'Melanoma')))
        not_melanoma_count = len(os.listdir(os.path.join(train_dir, 'NotMelanoma')))

        total = melanoma_count + not_melanoma_count

        # Calculate weights (higher weight for minority class)
        weight_melanoma = total / (2 * melanoma_count)
        weight_not_melanoma = total / (2 * not_melanoma_count)

        return {0: weight_melanoma, 1: weight_not_melanoma}

    def get_dataset_info(self, train_dir: Optional[str] = None,
                        val_dir: Optional[str] = None,
                        test_dir: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the dataset."""
        train_dir = train_dir or os.path.join(self.data_dir, 'training')
        val_dir = val_dir or os.path.join(self.data_dir, 'validation')
        test_dir = test_dir or os.path.join(self.data_dir, 'testing')

        def count_images(directory):
            melanoma = len(os.listdir(os.path.join(directory, 'Melanoma')))
            not_melanoma = len(os.listdir(os.path.join(directory, 'NotMelanoma')))
            return {'melanoma': melanoma, 'not_melanoma': not_melanoma, 'total': melanoma + not_melanoma}

        return {
            'training': count_images(train_dir),
            'validation': count_images(val_dir),
            'testing': count_images(test_dir),
            'input_shape': self.input_shape,
            'batch_size': self.batch_size,
            'augmentation_enabled': self.enable_augmentation,
        }


def create_data_generators(
    data_dir: str,
    input_shape: Tuple[int, int] = (299, 299),
    batch_size: int = 16,
    enable_augmentation: bool = True,
) -> Tuple:
    """
    Convenience function to create data generators.

    Returns:
        Tuple of (train_generator, val_generator, test_generator, class_weights)
    """
    loader = DataLoader(
        data_dir=data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        enable_augmentation=enable_augmentation,
    )

    train_gen, val_gen, test_gen = loader.create_generators()
    class_weights = loader.calculate_class_weights()

    return train_gen, val_gen, test_gen, class_weights
