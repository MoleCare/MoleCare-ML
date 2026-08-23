"""
ABCDE Melanoma Symptom Analyzer

Analyzes skin lesion images for the ABCDE criteria of melanoma:
- A: Asymmetry
- B: Border irregularity
- C: Color variation
- D: Diameter (estimated)
- E: Evolution (requires multiple images over time)

This module provides both traditional computer vision analysis
and ML-based analysis for comprehensive mole assessment.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ABCDEScore:
    """ABCDE analysis results."""
    asymmetry_score: float  # 0-1, higher = more asymmetric
    border_score: float     # 0-1, higher = more irregular
    color_score: float      # 0-1, higher = more variation
    diameter_mm: float      # Estimated diameter in mm
    evolution_score: Optional[float] = None  # Requires historical images

    # Individual risk assessments
    asymmetry_risk: str = ""
    border_risk: str = ""
    color_risk: str = ""
    diameter_risk: str = ""

    # Overall assessment
    total_score: float = 0.0
    risk_level: str = ""
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class ABCDEAnalyzer:
    """
    Analyzes skin lesions for ABCDE melanoma criteria.

    The ABCDE rule is a mnemonic used by dermatologists to identify
    potentially malignant melanomas:

    A - Asymmetry: One half doesn't match the other
    B - Border: Edges are irregular, ragged, or blurred
    C - Color: Multiple colors or uneven distribution
    D - Diameter: Larger than 6mm (pencil eraser size)
    E - Evolution: Changes in size, shape, or color over time

    Usage:
        analyzer = ABCDEAnalyzer()
        result = analyzer.analyze(image)
        print(f"Risk Level: {result.risk_level}")
        print(f"Recommendations: {result.recommendations}")
    """

    # Scoring thresholds
    ASYMMETRY_THRESHOLD = 0.3
    BORDER_THRESHOLD = 0.4
    COLOR_THRESHOLD = 0.35
    DIAMETER_THRESHOLD_MM = 6.0

    # Risk weights for total score calculation
    WEIGHTS = {
        'asymmetry': 0.25,
        'border': 0.25,
        'color': 0.30,
        'diameter': 0.20
    }

    def __init__(self, pixels_per_mm: float = 10.0):
        """
        Initialize analyzer.

        Args:
            pixels_per_mm: Conversion factor for diameter estimation.
                           Should be calibrated for your camera/setup.
        """
        self.pixels_per_mm = pixels_per_mm

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reference_mm: Optional[float] = None
    ) -> ABCDEScore:
        """
        Perform complete ABCDE analysis on a lesion image.

        Args:
            image: BGR image of the skin lesion
            mask: Optional binary mask of the lesion
            reference_mm: Optional reference size for calibration

        Returns:
            ABCDEScore with all analysis results
        """
        # Preprocess image
        image = self._preprocess_image(image)

        # Segment lesion if mask not provided
        if mask is None:
            mask = self._segment_lesion(image)

        # Analyze each criterion
        asymmetry_score = self._analyze_asymmetry(mask)
        border_score = self._analyze_border(mask)
        color_score = self._analyze_color(image, mask)
        diameter_mm = self._estimate_diameter(mask, reference_mm)

        # Assess risk for each criterion
        asymmetry_risk = self._assess_asymmetry_risk(asymmetry_score)
        border_risk = self._assess_border_risk(border_score)
        color_risk = self._assess_color_risk(color_score)
        diameter_risk = self._assess_diameter_risk(diameter_mm)

        # Calculate total score and risk level
        total_score = self._calculate_total_score(
            asymmetry_score, border_score, color_score, diameter_mm
        )
        risk_level = self._determine_risk_level(total_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            asymmetry_risk, border_risk, color_risk, diameter_risk, risk_level
        )

        return ABCDEScore(
            asymmetry_score=asymmetry_score,
            border_score=border_score,
            color_score=color_score,
            diameter_mm=diameter_mm,
            asymmetry_risk=asymmetry_risk,
            border_risk=border_risk,
            color_risk=color_risk,
            diameter_risk=diameter_risk,
            total_score=total_score,
            risk_level=risk_level,
            recommendations=recommendations
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis."""
        # Ensure correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Apply slight Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)

        return image

    def _segment_lesion(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the lesion from the background.

        Uses color-based segmentation with morphological operations.
        """
        # Convert to LAB color space (better for skin/lesion separation)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Apply Otsu's thresholding
        _, mask = cv2.threshold(
            l_channel, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find largest contour (assume it's the lesion)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        return mask

    def _analyze_asymmetry(self, mask: np.ndarray) -> float:
        """
        Analyze asymmetry of the lesion.

        Compares the lesion to its reflection along both axes.

        Returns:
            Asymmetry score 0-1 (0 = symmetric, 1 = highly asymmetric)
        """
        # Find bounding box
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        x, y, w, h = cv2.boundingRect(contours[0])
        lesion_roi = mask[y:y+h, x:x+w]

        # Pad to make square if needed
        max_dim = max(w, h)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        padded[pad_y:pad_y+h, pad_x:pad_x+w] = lesion_roi

        # Compare with horizontal flip
        h_flip = cv2.flip(padded, 1)
        h_diff = cv2.bitwise_xor(padded, h_flip)
        h_asymmetry = np.sum(h_diff > 0) / np.sum(padded > 0) if np.sum(padded > 0) > 0 else 0

        # Compare with vertical flip
        v_flip = cv2.flip(padded, 0)
        v_diff = cv2.bitwise_xor(padded, v_flip)
        v_asymmetry = np.sum(v_diff > 0) / np.sum(padded > 0) if np.sum(padded > 0) > 0 else 0

        # Average asymmetry
        asymmetry = (h_asymmetry + v_asymmetry) / 2

        return min(asymmetry, 1.0)

    def _analyze_border(self, mask: np.ndarray) -> float:
        """
        Analyze border irregularity.

        Uses fractal dimension and compactness metrics.

        Returns:
            Border irregularity score 0-1 (0 = smooth, 1 = very irregular)
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area == 0 or perimeter == 0:
            return 0.0

        # Compactness (circularity) - perfect circle = 1
        compactness = (4 * np.pi * area) / (perimeter ** 2)

        # Irregularity = 1 - compactness
        irregularity = 1 - compactness

        # Analyze border roughness using contour approximation
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        roughness = len(approx) / 20  # Normalize by expected smooth contour points

        # Combine metrics
        border_score = (irregularity * 0.6 + min(roughness, 1.0) * 0.4)

        return min(border_score, 1.0)

    def _analyze_color(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Analyze color variation within the lesion.

        Looks for multiple colors and uneven distribution.

        Returns:
            Color variation score 0-1 (0 = uniform, 1 = high variation)
        """
        # Extract lesion pixels
        lesion_pixels = image[mask > 0]

        if len(lesion_pixels) == 0:
            return 0.0

        # Convert to different color spaces for analysis
        lesion_lab = cv2.cvtColor(
            lesion_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3)

        # Calculate color statistics
        l_std = np.std(lesion_lab[:, 0])
        a_std = np.std(lesion_lab[:, 1])
        b_std = np.std(lesion_lab[:, 2])

        # Normalize standard deviations
        l_variation = min(l_std / 30, 1.0)  # L ranges 0-100
        a_variation = min(a_std / 40, 1.0)  # a ranges -128 to 127
        b_variation = min(b_std / 40, 1.0)  # b ranges -128 to 127

        # Count distinct color clusters using k-means
        n_colors = self._count_color_clusters(lesion_pixels)
        color_diversity = min((n_colors - 1) / 5, 1.0)  # Normalize: 6+ colors = 1.0

        # Weighted combination
        color_score = (
            l_variation * 0.3 +
            a_variation * 0.2 +
            b_variation * 0.2 +
            color_diversity * 0.3
        )

        return min(color_score, 1.0)

    def _count_color_clusters(
        self,
        pixels: np.ndarray,
        max_clusters: int = 8
    ) -> int:
        """Count distinct color clusters in the lesion."""
        from sklearn.cluster import KMeans

        if len(pixels) < max_clusters:
            return 1

        # Subsample for speed
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]

        # Try different numbers of clusters
        best_n = 1
        best_score = float('inf')

        for n in range(2, min(max_clusters, len(pixels) // 10) + 1):
            try:
                kmeans = KMeans(n_clusters=n, n_init=3, max_iter=100, random_state=42)
                kmeans.fit(pixels)
                # Use inertia (within-cluster sum of squares)
                score = kmeans.inertia_ / len(pixels)
                if score < best_score * 0.7:  # Significant improvement
                    best_score = score
                    best_n = n
            except Exception:
                break

        return best_n

    def _estimate_diameter(
        self,
        mask: np.ndarray,
        reference_mm: Optional[float] = None
    ) -> float:
        """
        Estimate the diameter of the lesion in millimeters.

        Args:
            mask: Binary mask of the lesion
            reference_mm: Optional calibration reference

        Returns:
            Estimated diameter in mm
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        diameter_pixels = radius * 2

        # Convert to mm
        if reference_mm:
            # Use provided reference for calibration
            return diameter_pixels / self.pixels_per_mm * reference_mm
        else:
            return diameter_pixels / self.pixels_per_mm

    def _assess_asymmetry_risk(self, score: float) -> str:
        """Assess risk level for asymmetry."""
        if score < 0.2:
            return "low"
        elif score < 0.35:
            return "moderate"
        elif score < 0.5:
            return "high"
        else:
            return "very_high"

    def _assess_border_risk(self, score: float) -> str:
        """Assess risk level for border irregularity."""
        if score < 0.25:
            return "low"
        elif score < 0.4:
            return "moderate"
        elif score < 0.6:
            return "high"
        else:
            return "very_high"

    def _assess_color_risk(self, score: float) -> str:
        """Assess risk level for color variation."""
        if score < 0.2:
            return "low"
        elif score < 0.35:
            return "moderate"
        elif score < 0.5:
            return "high"
        else:
            return "very_high"

    def _assess_diameter_risk(self, diameter_mm: float) -> str:
        """Assess risk level for diameter."""
        if diameter_mm < 4:
            return "low"
        elif diameter_mm < 6:
            return "moderate"
        elif diameter_mm < 10:
            return "high"
        else:
            return "very_high"

    def _calculate_total_score(
        self,
        asymmetry: float,
        border: float,
        color: float,
        diameter_mm: float
    ) -> float:
        """Calculate weighted total ABCDE score."""
        # Normalize diameter to 0-1 scale
        diameter_normalized = min(diameter_mm / 12, 1.0)

        total = (
            asymmetry * self.WEIGHTS['asymmetry'] +
            border * self.WEIGHTS['border'] +
            color * self.WEIGHTS['color'] +
            diameter_normalized * self.WEIGHTS['diameter']
        )

        return min(total, 1.0)

    def _determine_risk_level(self, total_score: float) -> str:
        """Determine overall risk level from total score."""
        if total_score < 0.25:
            return "low"
        elif total_score < 0.45:
            return "moderate"
        elif total_score < 0.65:
            return "high"
        else:
            return "very_high"

    def _generate_recommendations(
        self,
        asymmetry_risk: str,
        border_risk: str,
        color_risk: str,
        diameter_risk: str,
        overall_risk: str
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # General recommendation based on overall risk
        if overall_risk == "very_high":
            recommendations.append(
                "URGENT: This lesion shows multiple warning signs. "
                "Please consult a dermatologist as soon as possible."
            )
        elif overall_risk == "high":
            recommendations.append(
                "This lesion shows concerning features. "
                "We recommend scheduling a dermatologist appointment within 2 weeks."
            )
        elif overall_risk == "moderate":
            recommendations.append(
                "Some features warrant monitoring. "
                "Consider a dermatologist consultation if changes occur."
            )
        else:
            recommendations.append(
                "This lesion appears low risk, but continue to monitor for changes."
            )

        # Specific recommendations
        if asymmetry_risk in ["high", "very_high"]:
            recommendations.append(
                "A (Asymmetry): The lesion shows significant asymmetry. "
                "Melanomas often grow unevenly."
            )

        if border_risk in ["high", "very_high"]:
            recommendations.append(
                "B (Border): The border appears irregular or poorly defined. "
                "Smooth, even borders are typically benign."
            )

        if color_risk in ["high", "very_high"]:
            recommendations.append(
                "C (Color): Multiple colors detected within the lesion. "
                "Uniform color is generally a good sign."
            )

        if diameter_risk in ["high", "very_high"]:
            recommendations.append(
                "D (Diameter): The lesion is larger than 6mm. "
                "Larger lesions should be evaluated."
            )

        # Self-monitoring advice
        recommendations.append(
            "E (Evolution): Take photos monthly to track any changes in "
            "size, shape, or color. Report any changes to your doctor."
        )

        return recommendations


def analyze_mole_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    time_delta_days: int = 30
) -> Dict[str, Any]:
    """
    Compare two images of the same mole taken at different times.

    Args:
        image1: Earlier image (BGR)
        image2: Later image (BGR)
        time_delta_days: Time between images

    Returns:
        Comparison results with evolution assessment
    """
    analyzer = ABCDEAnalyzer()

    # Analyze both images
    result1 = analyzer.analyze(image1)
    result2 = analyzer.analyze(image2)

    # Calculate changes
    changes = {
        'asymmetry_change': result2.asymmetry_score - result1.asymmetry_score,
        'border_change': result2.border_score - result1.border_score,
        'color_change': result2.color_score - result1.color_score,
        'diameter_change_mm': result2.diameter_mm - result1.diameter_mm,
        'total_score_change': result2.total_score - result1.total_score,
    }

    # Assess evolution risk
    significant_changes = []

    if abs(changes['asymmetry_change']) > 0.1:
        significant_changes.append('asymmetry')
    if abs(changes['border_change']) > 0.1:
        significant_changes.append('border')
    if abs(changes['color_change']) > 0.1:
        significant_changes.append('color')
    if changes['diameter_change_mm'] > 1.0:
        significant_changes.append('size increase')

    # Evolution score
    evolution_score = len(significant_changes) / 4

    # Generate evolution assessment
    if evolution_score > 0.5 or changes['total_score_change'] > 0.15:
        evolution_risk = "high"
        evolution_message = (
            f"SIGNIFICANT CHANGES DETECTED in {time_delta_days} days. "
            f"Changed features: {', '.join(significant_changes)}. "
            "Please consult a dermatologist promptly."
        )
    elif evolution_score > 0.25 or changes['total_score_change'] > 0.08:
        evolution_risk = "moderate"
        evolution_message = (
            f"Some changes observed over {time_delta_days} days. "
            "Continue monitoring and consider consultation."
        )
    else:
        evolution_risk = "low"
        evolution_message = (
            f"No significant changes in {time_delta_days} days. "
            "Continue regular monitoring."
        )

    return {
        'earlier_analysis': result1,
        'later_analysis': result2,
        'changes': changes,
        'evolution_score': evolution_score,
        'evolution_risk': evolution_risk,
        'evolution_message': evolution_message,
        'significant_changes': significant_changes,
        'time_delta_days': time_delta_days,
    }
