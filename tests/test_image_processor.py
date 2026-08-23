"""Tests for ImageProcessor.

Fixtures here are generated synthetically at runtime. Never commit a real
skin photograph to this repository — not as a file, and not as an embedded
base64 string in a test.
"""

import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from ml_model_serving.image_processor import ImageProcessor


def synthetic_lesion_image(size=(1024, 1024), mole_fraction=0.4, mode="RGB",
                           fmt="JPEG"):
    """Build a fake 'skin with a mole' image: warm background, dark ellipse.

    Deterministic — no randomness — so failures are reproducible.
    """
    w, h = size
    img = Image.new(mode if mode != "L" else "RGB", size, (222, 184, 156))
    draw = ImageDraw.Draw(img)

    radius_x = int(w * mole_fraction / 2)
    radius_y = int(h * mole_fraction / 2)
    cx, cy = w // 2, h // 2
    draw.ellipse(
        [cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y],
        fill=(72, 48, 40),
    )

    if mode != "RGB":
        img = img.convert(mode)

    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def as_base64(image_bytes, data_uri=False, mime="image/jpeg"):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}" if data_uri else encoded


class ImageProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()

    def test_prepare_input_image_returns_model_ready_batch(self):
        payload = as_base64(synthetic_lesion_image())

        result = self.processor.prepare_input_image(payload)

        expected_size = ImageProcessor.XCEPTION_INPUT_SHAPE_SIZE
        self.assertEqual(result.shape, (1, expected_size, expected_size, 3))
        self.assertEqual(result.dtype, np.float16)
        # Xception preprocessing scales pixels into [-1, 1]
        self.assertGreaterEqual(float(result.min()), -1.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_accepts_data_uri_prefix(self):
        payload = as_base64(synthetic_lesion_image(), data_uri=True)

        result = self.processor.prepare_input_image(payload)

        expected_size = ImageProcessor.XCEPTION_INPUT_SHAPE_SIZE
        self.assertEqual(result.shape, (1, expected_size, expected_size, 3))

    def test_converts_non_rgb_images(self):
        payload = as_base64(synthetic_lesion_image(mode="L"))

        result = self.processor.prepare_input_image(payload)

        self.assertEqual(result.shape[-1], 3, "grayscale input should become RGB")

    def test_accepts_png(self):
        payload = as_base64(synthetic_lesion_image(fmt="PNG"), data_uri=True,
                            mime="image/png")

        result = self.processor.prepare_input_image(payload)

        self.assertEqual(result.shape[0], 1)

    def test_prepare_input_from_bytes_matches_base64_path(self):
        raw = synthetic_lesion_image()

        from_bytes = self.processor.prepare_input_from_bytes(raw)
        from_b64 = self.processor.prepare_input_image(as_base64(raw))

        np.testing.assert_array_equal(from_bytes, from_b64)

    def test_rejects_invalid_base64(self):
        with self.assertRaises(ValueError):
            self.processor.prepare_input_image("not-base64-at-all!!")

    def test_rejects_non_image_payload(self):
        payload = base64.b64encode(b"this is plain text").decode("utf-8")

        with self.assertRaises(ValueError):
            self.processor.prepare_input_image(payload)

    def test_rejects_oversized_image(self):
        oversized = b"\xff\xd8" + b"\x00" * (
            (ImageProcessor.MAX_IMAGE_SIZE_MB + 1) * 1024 * 1024
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor.prepare_input_image(as_base64(oversized))

        self.assertIn("too large", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
