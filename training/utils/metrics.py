"""
Medical Metrics for Melanoma Classification

Provides specialized metrics for medical classification tasks:
- Sensitivity (Recall): Ability to detect melanoma (True Positive Rate)
- Specificity: Ability to correctly identify non-melanoma (True Negative Rate)
- AUC-ROC: Area under the ROC curve
- PPV/NPV: Positive/Negative Predictive Values

For melanoma detection, high SENSITIVITY is critical because
missing a melanoma (false negative) is more dangerous than
a false alarm (false positive).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MedicalMetrics:
    """
    Calculate and track medical classification metrics.

    For melanoma detection:
    - Class 0: Melanoma (Positive)
    - Class 1: NotMelanoma (Negative)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def update(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Update with batch predictions."""
        self.y_true.extend(y_true.flatten())
        self.y_prob.extend(y_prob.flatten())
        self.y_pred.extend((y_prob.flatten() >= self.threshold).astype(int))

    def reset(self):
        """Reset accumulated predictions."""
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def calculate(self) -> Dict[str, float]:
        """
        Calculate all metrics.

        Returns dict with:
        - accuracy: Overall accuracy
        - auc: Area under ROC curve
        - sensitivity: True Positive Rate (Recall for melanoma)
        - specificity: True Negative Rate
        - ppv: Positive Predictive Value (Precision for melanoma)
        - npv: Negative Predictive Value
        - f1: F1 score
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_prob = np.array(self.y_prob)

        # For binary classification with Melanoma=0, NotMelanoma=1
        # We need to invert to calculate metrics correctly
        # Melanoma is the "positive" class we want to detect

        # Confusion matrix
        # Note: sklearn expects [TN, FP, FN, TP] when positive class is first
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Positive Predictive Value (Precision) = TP / (TP + FP)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Negative Predictive Value = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # AUC (need to invert probabilities for Melanoma=positive)
        try:
            auc = roc_auc_score(1 - y_true, 1 - y_prob)
        except ValueError:
            auc = 0.0

        # F1 Score
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,  # Most important for melanoma
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }

    def find_optimal_threshold(self) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold using Youden's J statistic.

        J = Sensitivity + Specificity - 1

        For melanoma detection, you might want to prioritize sensitivity
        over specificity, in which case use a lower threshold.
        """
        y_true = np.array(self.y_true)
        y_prob = np.array(self.y_prob)

        # Invert for Melanoma=positive
        fpr, tpr, thresholds = roc_curve(1 - y_true, 1 - y_prob)

        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = 1 - thresholds[optimal_idx]  # Invert back

        # Calculate metrics at optimal threshold
        self.threshold = optimal_threshold
        self.y_pred = [(p >= optimal_threshold) for p in self.y_prob]
        metrics = self.calculate()

        return optimal_threshold, metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self) -> str:
        """Get sklearn classification report."""
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=['Melanoma', 'NotMelanoma'],
        )


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate all medical metrics for predictions.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    metrics = MedicalMetrics(threshold)
    metrics.update(y_true, y_prob)
    return metrics.calculate()


def create_keras_metrics() -> list:
    """
    Create list of Keras metrics for model compilation.

    Returns comprehensive metrics for medical classification.
    """
    return [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='sensitivity'),  # Recall = Sensitivity
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
    ]


class SpecificityMetric(tf.keras.metrics.Metric):
    """Custom Keras metric for Specificity."""

    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # For Melanoma=0, NotMelanoma=1
        # TN: predicted 1, actual 1
        # FP: predicted 0, actual 1
        tn = tf.reduce_sum((1 - y_pred) * (1 - y_true))
        fp = tf.reduce_sum(y_pred * (1 - y_true))

        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        denominator = self.true_negatives + self.false_positives
        return tf.cond(
            denominator > 0,
            lambda: self.true_negatives / denominator,
            lambda: 0.0
        )

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
