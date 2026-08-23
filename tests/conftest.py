"""Shared test fixtures.

The serving package imports tensorflow, cv2, scikit-learn and huggingface_hub.
Installing all of those to run a 400-response assertion is not worth it, so any
that are genuinely missing get a stub installed here.

The stubs engage ONLY when the real library cannot be imported. CI installs
requirements.lock, so CI exercises the real tensorflow; a developer with a bare
venv still gets a fast, meaningful test run. That keeps one suite honest in both
places instead of silently testing the stubs everywhere.
"""

import base64
import importlib
import io
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _missing(name):
    try:
        importlib.import_module(name)
        return False
    except Exception:
        return True


# --------------------------------------------------------------------------
# tensorflow: prepare_input_image() calls tf.image.resize and the Xception
# preprocessor, so a bare MagicMock is not enough -- those two must return real
# arrays or the pipeline breaks on .astype().
# --------------------------------------------------------------------------
if _missing("tensorflow"):
    from PIL import Image as _PILImage

    def _resize(arr, size):
        a = np.asarray(arr, dtype=np.float32)
        img = _PILImage.fromarray(np.clip(a, 0, 255).astype("uint8"))
        img = img.resize((int(size[1]), int(size[0])))
        return np.asarray(img, dtype=np.float32)

    def _xception_preprocess(x):
        # Xception scales into [-1, 1]; matching it keeps stubbed and real runs
        # numerically comparable.
        return np.asarray(x, dtype=np.float32) / 127.5 - 1.0

    tf = MagicMock(name="tensorflow")
    tf.image.resize = _resize
    tf.keras.applications.xception.preprocess_input = _xception_preprocess
    sys.modules["tensorflow"] = tf

for _name in ("cv2", "sklearn", "sklearn.preprocessing", "sklearn.metrics", "huggingface_hub"):
    if _missing(_name):
        sys.modules[_name] = MagicMock(name=_name)


RAW_NOT_MELANOMA = 0.87  # model's raw output = P(NotMelanoma)


class _FakeModel:
    """Stands in for the loaded Xception SavedModel."""

    def __init__(self, raw=RAW_NOT_MELANOMA):
        self.raw = raw
        self.calls = []

    def predict(self, x, verbose=0):
        self.calls.append(x)
        return np.array([[self.raw]], dtype=np.float32)


@pytest.fixture
def fake_model():
    """Install a deterministic model, bypassing any real SavedModel load."""
    from ml_model_serving.model_prediction_service import ModelPredictionService

    previous_model = ModelPredictionService._model
    previous_instance = ModelPredictionService._instance
    model = _FakeModel()
    ModelPredictionService._model = model
    ModelPredictionService._instance = object.__new__(ModelPredictionService)
    yield model
    ModelPredictionService._model = previous_model
    ModelPredictionService._instance = previous_instance


@pytest.fixture
def app(fake_model):
    from ml_model_serving import app as flask_app

    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _png_bytes(size=(64, 64), colour=(180, 120, 110)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_b64():
    """A real, decodable PNG -- not a placeholder string."""
    return base64.b64encode(_png_bytes()).decode()


@pytest.fixture
def predict_body(image_b64):
    return {"predictionid": "3f2504e0-4f89-11d3-9a0c-0305e82c3301", "imagebase64": image_b64}
