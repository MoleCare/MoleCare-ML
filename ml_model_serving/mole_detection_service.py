"""
Mole Detection and Extraction Service

Provides functionality to detect moles in images, extract bounding boxes,
and return cropped mole images for analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedMole:
    """Represents a detected mole in an image."""
    id: int
    bounding_box: dict  # {"x": int, "y": int, "width": int, "height": int}
    center: dict  # {"x": float, "y": float}
    area_pixels: float
    contour: List[List[int]]  # List of [x, y] points
    confidence: float  # Detection confidence 0-1
    is_primary: bool = False  # True if this is the main/largest mole

    def to_dict(self):
        return {
            'id': self.id,
            'bounding_box': self.bounding_box,
            'center': self.center,
            'area_pixels': self.area_pixels,
            'contour': self.contour,
            'confidence': self.confidence,
            'is_primary': self.is_primary
        }


@dataclass
class DetectionResult:
    """Result of mole detection operation."""
    success: bool
    moles: List[DetectedMole] = field(default_factory=list)
    total_count: int = 0
    primary_mole_id: Optional[int] = None
    cropped_mole_base64: Optional[str] = None  # Primary mole cropped image
    mask_base64: Optional[str] = None  # Segmentation mask
    quality_warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self):
        return {
            'success': self.success,
            'moles': [m.to_dict() for m in self.moles],
            'total_count': self.total_count,
            'primary_mole_id': self.primary_mole_id,
            'cropped_mole_base64': self.cropped_mole_base64,
            'mask_base64': self.mask_base64,
            'quality_warnings': self.quality_warnings,
            'error': self.error
        }


class MoleDetectionService:
    """Service for detecting and extracting moles from images."""

    # Detection parameters
    MIN_MOLE_AREA = 100  # Minimum area in pixels to be considered a mole
    MAX_MOLE_AREA_RATIO = 0.8  # Max ratio of image area a mole can occupy
    MIN_MOLE_AREA_RATIO = 0.001  # Min ratio of image area for a mole

    # Morphological operation kernel sizes
    MORPH_KERNEL_SIZE = 5
    BLUR_KERNEL_SIZE = 5

    # Color thresholds for skin lesion detection
    DARKNESS_THRESHOLD = 0.7  # How dark lesions should be relative to skin

    def __init__(self):
        self.image_processor = None  # Lazy import to avoid circular dependency

    def _get_image_processor(self):
        """Lazy load image processor."""
        if self.image_processor is None:
            from .ImageProcessor import ImageProcessor
            self.image_processor = ImageProcessor()
        return self.image_processor

    def detect_moles(self, image_base64: str,
                     return_mask: bool = False,
                     return_cropped: bool = True,
                     crop_padding: int = 20) -> DetectionResult:
        """
        Detect moles in an image and return their locations.

        Args:
            image_base64: Base64 encoded image
            return_mask: Whether to return the segmentation mask
            return_cropped: Whether to return cropped primary mole image
            crop_padding: Padding around mole when cropping (pixels)

        Returns:
            DetectionResult with detected moles and optional outputs
        """
        result = DetectionResult(success=False)

        try:
            # Load image
            processor = self._get_image_processor()
            bgr_image, pil_image, quality_report = processor.load_image_for_analysis(image_base64)

            # Add quality warnings
            result.quality_warnings = quality_report.warnings

            # Get image dimensions
            height, width = bgr_image.shape[:2]
            image_area = width * height

            # Segment lesions
            mask, contours = self._segment_lesions(bgr_image)

            if len(contours) == 0:
                result.success = True
                result.quality_warnings.append("No moles detected. Ensure the mole is clearly visible and well-lit.")
                return result

            # Process each contour
            detected_moles = []
            largest_area = 0
            primary_id = None

            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Filter by area
                area_ratio = area / image_area
                if area_ratio < self.MIN_MOLE_AREA_RATIO:
                    continue
                if area_ratio > self.MAX_MOLE_AREA_RATIO:
                    continue
                if area < self.MIN_MOLE_AREA:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Get center
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w / 2, y + h / 2

                # Calculate confidence based on shape regularity
                confidence = self._calculate_detection_confidence(contour, area)

                # Convert contour to list format
                contour_points = contour.squeeze().tolist()
                if isinstance(contour_points[0], int):
                    contour_points = [contour_points]

                mole = DetectedMole(
                    id=idx,
                    bounding_box={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    center={"x": float(cx), "y": float(cy)},
                    area_pixels=float(area),
                    contour=contour_points,
                    confidence=confidence
                )

                # Track largest mole
                if area > largest_area:
                    largest_area = area
                    primary_id = idx

                detected_moles.append(mole)

            # Mark primary mole
            for mole in detected_moles:
                if mole.id == primary_id:
                    mole.is_primary = True

            result.moles = detected_moles
            result.total_count = len(detected_moles)
            result.primary_mole_id = primary_id
            result.success = True

            # Generate cropped image of primary mole
            if return_cropped and primary_id is not None:
                primary_mole = next((m for m in detected_moles if m.id == primary_id), None)
                if primary_mole:
                    cropped = self._crop_mole(bgr_image, primary_mole, crop_padding)
                    result.cropped_mole_base64 = processor.encode_image_to_base64(cropped)

            # Generate mask image
            if return_mask:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                result.mask_base64 = processor.encode_image_to_base64(mask_colored)

        except Exception as e:
            logger.error(f"Mole detection failed: {e}")
            result.error = str(e)

        return result

    def _segment_lesions(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Segment skin lesions from the image.

        Uses multiple techniques:
        1. LAB color space thresholding
        2. Adaptive thresholding
        3. Morphological operations

        Args:
            bgr_image: Input image in BGR format

        Returns:
            Tuple of (binary mask, list of contours)
        """
        # Convert to LAB color space (better for skin lesion detection)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply Gaussian blur to reduce noise
        l_blurred = cv2.GaussianBlur(l_channel, (self.BLUR_KERNEL_SIZE, self.BLUR_KERNEL_SIZE), 0)

        # Otsu's thresholding on L channel (luminance)
        _, otsu_mask = cv2.threshold(l_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Also try adaptive thresholding for better edge detection
        adaptive_mask = cv2.adaptiveThreshold(
            l_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # Combine masks
        combined_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE)
        )

        # Close small gaps
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill holes
        mask = self._fill_holes(mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in the binary mask."""
        # Copy the mask
        filled = mask.copy()

        # Create a mask slightly larger than the image
        h, w = mask.shape[:2]
        fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood fill from the corners
        cv2.floodFill(filled, fill_mask, (0, 0), 255)

        # Invert the flood-filled image
        filled_inv = cv2.bitwise_not(filled)

        # Combine with original mask
        return cv2.bitwise_or(mask, filled_inv)

    def _calculate_detection_confidence(self, contour: np.ndarray, area: float) -> float:
        """
        Calculate confidence score for detected mole based on shape characteristics.

        Args:
            contour: Contour points
            area: Contour area

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5  # Base confidence

        # Check circularity (moles tend to be roughly circular)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            # Perfect circle = 1.0
            if circularity > 0.5:
                confidence += 0.2
            elif circularity > 0.3:
                confidence += 0.1

        # Check solidity (how "filled in" the shape is)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.8:
                confidence += 0.2
            elif solidity > 0.6:
                confidence += 0.1

        # Size factor (very small or very large = less confidence)
        if 500 < area < 100000:
            confidence += 0.1

        return min(1.0, confidence)

    def _crop_mole(self, image: np.ndarray, mole: DetectedMole, padding: int) -> np.ndarray:
        """
        Crop the mole region from the image with padding.

        Args:
            image: Source image
            mole: Detected mole object
            padding: Padding around the mole

        Returns:
            Cropped image
        """
        height, width = image.shape[:2]
        bb = mole.bounding_box

        # Calculate crop region with padding
        x1 = max(0, bb["x"] - padding)
        y1 = max(0, bb["y"] - padding)
        x2 = min(width, bb["x"] + bb["width"] + padding)
        y2 = min(height, bb["y"] + bb["height"] + padding)

        return image[y1:y2, x1:x2].copy()

    def extract_and_enhance_mole(self, image_base64: str,
                                  mole_id: Optional[int] = None) -> DetectionResult:
        """
        Extract a specific mole and enhance the image for better analysis.

        Args:
            image_base64: Base64 encoded image
            mole_id: ID of mole to extract (None = primary/largest)

        Returns:
            DetectionResult with enhanced cropped mole
        """
        # First detect all moles
        result = self.detect_moles(image_base64, return_mask=False, return_cropped=False)

        if not result.success or result.total_count == 0:
            return result

        # Select mole to extract
        target_id = mole_id if mole_id is not None else result.primary_mole_id
        target_mole = next((m for m in result.moles if m.id == target_id), None)

        if target_mole is None:
            result.error = f"Mole with ID {target_id} not found"
            return result

        try:
            processor = self._get_image_processor()
            bgr_image, _, _ = processor.load_image_for_analysis(image_base64)

            # Crop with extra padding
            cropped = self._crop_mole(bgr_image, target_mole, padding=30)

            # Enhance the cropped image
            enhanced = self._enhance_mole_image(cropped)

            result.cropped_mole_base64 = processor.encode_image_to_base64(enhanced)

        except Exception as e:
            logger.error(f"Mole extraction failed: {e}")
            result.error = str(e)

        return result

    def _enhance_mole_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance mole image for better visibility and analysis.

        Args:
            image: Cropped mole image (BGR)

        Returns:
            Enhanced image
        """
        # Convert to LAB for better color enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Light sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) / 1.0
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.3 + np.eye(3) * 0.7)

        return sharpened


# Singleton instance
_detection_service = None


def get_detection_service() -> MoleDetectionService:
    """Get singleton instance of MoleDetectionService."""
    global _detection_service
    if _detection_service is None:
        _detection_service = MoleDetectionService()
    return _detection_service
