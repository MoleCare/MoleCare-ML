"""Unit tests for DermFoundationService unavailable / ready contracts."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / "ml_model_serving" / "derm_foundation_service.py"


def _load_service_module():
    """Load DermFoundationService.py without importing flask via package __init__."""
    pkg = ModuleType("ml_model_serving")
    pkg.__path__ = [str(ROOT / "ml_model_serving")]
    sys.modules["ml_model_serving"] = pkg

    sys.modules.setdefault("tensorflow", MagicMock())
    sys.modules.setdefault("PIL", MagicMock())
    sys.modules.setdefault("PIL.Image", MagicMock())
    sys.modules.setdefault("numpy", np)

    spec = importlib.util.spec_from_file_location(
        "ml_model_serving.derm_foundation_service", SERVICE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ml_model_serving.derm_foundation_service"] = mod
    spec.loader.exec_module(mod)
    return mod


class DermFoundationServiceTests(unittest.TestCase):
    def setUp(self):
        for key in list(sys.modules):
            if key.startswith("ml_model_serving"):
                sys.modules.pop(key, None)
        self.mod = _load_service_module()
        self.mod.DERM_FOUNDATION_AVAILABLE = False
        self.mod.DERM_MODEL_LOADED = False
        self.mod.DermFoundationService._instance = None
        self.mod.DermFoundationService._derm_model = None
        self.mod.DermFoundationService._infer_fn = None
        self.mod.DermFoundationService._classifier = None
        self.mod.DermFoundationService._scaler = None

    def tearDown(self):
        for key in list(sys.modules):
            if key.startswith("ml_model_serving"):
                sys.modules.pop(key, None)

    def test_unavailable_when_classifier_missing(self):
        cls = self.mod.DermFoundationService
        self.mod.DERM_MODEL_LOADED = True
        self.mod.DERM_FOUNDATION_AVAILABLE = False
        cls._classifier = None
        cls._scaler = None
        self.assertFalse(cls.is_available())

        svc = object.__new__(cls)
        payload = svc.predict(b"not-an-image")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["code"], "model_unavailable")
        self.assertEqual(payload["model_type"], "derm_foundation")

    def test_unavailable_payload_shape(self):
        payload = self.mod.DermFoundationService.unavailable_payload("missing artifacts")
        self.assertEqual(payload["code"], "model_unavailable")
        self.assertIn("hint", payload)
        self.assertEqual(payload["error"], "missing artifacts")

    def test_predict_with_mocked_classifier(self):
        cls = self.mod.DermFoundationService
        self.mod.DERM_MODEL_LOADED = True
        self.mod.DERM_FOUNDATION_AVAILABLE = True

        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((1, 8))
        classifier = MagicMock()
        classifier.predict.return_value = np.array([[0.9]])

        cls._scaler = scaler
        cls._classifier = classifier

        with patch.object(cls, "_preprocess_image", classmethod(lambda c, data: b"png")), patch.object(
            cls, "_get_embedding", classmethod(lambda c, data: np.zeros(8, dtype=np.float32))
        ):
            svc = object.__new__(cls)
            result = svc.predict(b"img", threshold=0.5)

        self.assertTrue(result["success"])
        self.assertEqual(result["prediction"], "NotMelanoma")
        self.assertGreaterEqual(result["melanoma_probability"], 0.0)
        self.assertLessEqual(result["melanoma_probability"], 1.0)
        self.assertEqual(result["model_type"], "derm_foundation")


if __name__ == "__main__":
    unittest.main()
