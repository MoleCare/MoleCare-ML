import tensorflow as tf
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images for melanoma prediction model."""

    XCEPTION_INPUT_SHAPE_SIZE = 299
    MAX_IMAGE_SIZE_MB = 10
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'JPG', 'WEBP'}

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
