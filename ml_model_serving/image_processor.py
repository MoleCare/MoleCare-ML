import base64
import logging
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


class ImageQualityReport:
    """Report on image quality for mole analysis."""

    def __init__(self):
        self.is_valid = True
        self.warnings = []
        self.errors = []
        self.original_width = 0
        self.original_height = 0
        self.format = None
        self.resolution_ok = True
        self.framing_ok = True
        self.quality_score = 1.0  # 0-1 score

    def to_dict(self):
        return {
            'is_valid': self.is_valid,
            'warnings': self.warnings,
            'errors': self.errors,
            'original_width': self.original_width,
            'original_height': self.original_height,
            'format': self.format,
            'resolution_ok': self.resolution_ok,
            'framing_ok': self.framing_ok,
            'quality_score': self.quality_score
        }


class ImageProcessor:
    """Process images for melanoma prediction model."""

    XCEPTION_INPUT_SHAPE_SIZE = 299
    EFFICIENTNET_INPUT_SHAPE_SIZE = 224
    MAX_IMAGE_SIZE_MB = 10
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'JPG', 'WEBP'}

    # Photo size recommendations
    MIN_RESOLUTION = 640  # Minimum 640x640 for good analysis
    OPTIMAL_RESOLUTION = 1024  # Optimal 1024x1024
    MAX_RESOLUTION = 4096  # Max to prevent memory issues

    # Framing recommendations (mole should fill 60-80% of frame)
    OPTIMAL_MOLE_COVERAGE_MIN = 0.20  # At least 20% of image should be mole
    OPTIMAL_MOLE_COVERAGE_MAX = 0.80  # No more than 80%

    def prepare_input_image(self, image_base64):
        """
        Process base64 image entirely in-memory (no disk I/O).

        Args:
            image_base64: Base64 encoded image string, optionally with data URI prefix

        Returns:
            numpy array ready for model prediction

        Raises:
            ValueError: If image is invalid, too large, or unsupported format
        """
        # Strip data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            logger.error(f"Failed to decode base64: {e}")
            raise ValueError("Invalid base64 image data")

        # Validate image size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            raise ValueError(f"Image too large: {size_mb:.2f}MB (max {self.MAX_IMAGE_SIZE_MB}MB)")

        # Load image in-memory using PIL
        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise ValueError("Invalid image data")

        # Validate format
        if img.format and img.format.upper() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {img.format}")

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert PIL Image to numpy array
        input_image = np.array(img, dtype=np.float32)

        # Resize to model input size
        input_image = tf.image.resize(
            input_image,
            [self.XCEPTION_INPUT_SHAPE_SIZE, self.XCEPTION_INPUT_SHAPE_SIZE]
        )

        # Apply Xception preprocessing
        input_image = tf.keras.applications.xception.preprocess_input(input_image)

        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)

        # Cast to float16 for TF Serving compatibility
        input_image = input_image.astype('float16')

        return input_image

    def prepare_input_from_bytes(self, image_bytes):
        """
        Process raw image bytes (for Lambda container usage).

        Args:
            image_bytes: Raw image bytes

        Returns:
            numpy array ready for model prediction
        """
        return self.prepare_input_image(base64.b64encode(image_bytes).decode('utf-8'))

    def validate_image_quality(self, image_base64):
        """
        Validate image quality and provide recommendations.

        Args:
            image_base64: Base64 encoded image string

        Returns:
            ImageQualityReport with validation results and recommendations
        """
        report = ImageQualityReport()

        # Strip data URI prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            report.is_valid = False
            report.errors.append("Invalid base64 image data")
            return report

        # Validate file size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            report.is_valid = False
            report.errors.append(f"Image too large: {size_mb:.2f}MB (max {self.MAX_IMAGE_SIZE_MB}MB)")
            return report

        # Load image
        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            report.is_valid = False
            report.errors.append("Invalid image data - cannot open image")
            return report

        # Get image properties
        report.original_width = img.width
        report.original_height = img.height
        report.format = img.format

        # Validate format
        if img.format and img.format.upper() not in self.SUPPORTED_FORMATS:
            report.is_valid = False
            report.errors.append(f"Unsupported format: {img.format}. Use JPEG, PNG, or WEBP.")
            return report

        # Check resolution
        min_dimension = min(img.width, img.height)
        max_dimension = max(img.width, img.height)

        if min_dimension < self.MIN_RESOLUTION:
            report.resolution_ok = False
            report.warnings.append(
                f"Low resolution: {img.width}x{img.height}. "
                f"Minimum recommended: {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION} pixels."
            )
            report.quality_score -= 0.3

        if max_dimension > self.MAX_RESOLUTION:
            report.warnings.append(
                f"Very high resolution: {img.width}x{img.height}. "
                f"Image will be resized for processing."
            )

        # Check aspect ratio (should be close to square for centered mole)
        aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else 0
        if aspect_ratio > 2.0:
            report.warnings.append(
                f"Image is not square (aspect ratio: {aspect_ratio:.2f}). "
                "For best results, use a square crop centered on the mole."
            )
            report.quality_score -= 0.1

        # Optimal resolution bonus
        if min_dimension >= self.OPTIMAL_RESOLUTION:
            report.quality_score = min(1.0, report.quality_score + 0.1)

        # Ensure quality score is in valid range
        report.quality_score = max(0.0, min(1.0, report.quality_score))

        return report

    def load_image_for_analysis(self, image_base64, target_size=None):
        """
        Load image for analysis without model-specific preprocessing.
        Useful for mole detection and segmentation.

        Args:
            image_base64: Base64 encoded image string
            target_size: Optional tuple (width, height) to resize to

        Returns:
            tuple: (numpy array in BGR format for OpenCV, PIL Image, ImageQualityReport)
        """
        import cv2

        # Validate quality first
        report = self.validate_image_quality(image_base64)
        if not report.is_valid:
            raise ValueError(report.errors[0] if report.errors else "Invalid image")

        # Strip data URI prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode and load
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(BytesIO(image_bytes))

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array (RGB)
        np_image = np.array(pil_image)

        # Resize if target size specified
        if target_size:
            np_image = cv2.resize(np_image, target_size, interpolation=cv2.INTER_AREA)
            pil_image = Image.fromarray(np_image)

        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        return bgr_image, pil_image, report

    def encode_image_to_base64(self, image, format='JPEG', quality=90):
        """
        Encode image to base64 string.

        Args:
            image: numpy array (BGR) or PIL Image
            format: Output format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)

        Returns:
            Base64 encoded string
        """
        import cv2

        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image

        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Save to bytes
        buffer = BytesIO()
        if format.upper() == 'JPEG':
            pil_image.save(buffer, format='JPEG', quality=quality)
        else:
            pil_image.save(buffer, format=format)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
