"""Regression tests for module naming and model loading.

Both of these broke silently once: the PEP 8 rename left stale CamelCase
imports behind, and the Keras 3 upgrade made the bundled Keras 2.9 SavedModel
unloadable. Neither showed up in CI because nothing imported these paths.
"""

import importlib
import os
import unittest


class ModuleNamingTestCase(unittest.TestCase):
    """Every service module must import cleanly under its snake_case name."""

    MODULES = [
        "ml_model_serving.abcde_analyzer",
        "ml_model_serving.evolution_analysis_service",
        "ml_model_serving.image_processor",
        "ml_model_serving.mole_analysis_service",
        "ml_model_serving.mole_detection_service",
        "ml_model_serving.model_prediction_service",
        "ml_model_serving.validator",
    ]

    def test_all_service_modules_import(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_no_stale_camelcase_imports(self):
        """Guards against reintroducing `from .ImageProcessor import ...`."""
        import pathlib
        import re

        stale = re.compile(
            r"from\s+\.?(ABCDEAnalyzer|ImageProcessor|ModelPredictionService"
            r"|PredictionController)\s+import"
        )
        root = pathlib.Path(__file__).resolve().parent.parent / "ml_model_serving"
        offenders = [
            f"{path.name}:{i}"
            for path in root.glob("*.py")
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if stale.search(line)
        ]
        self.assertEqual(offenders, [], f"stale CamelCase imports: {offenders}")


class ModelLoadingTestCase(unittest.TestCase):
    """The bundled SavedModel must actually load with the pinned stack."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.environ.get("MODEL_PATH", "./cnn-models/xception/1")
        if not os.path.isdir(cls.model_path):
            raise unittest.SkipTest(
                f"model not present at {cls.model_path} - download it from Releases"
            )

    def test_tf_keras_is_available(self):
        """Keras 3 cannot read the Keras 2.9 SavedModel; tf_keras can."""
        import ml_model_serving.model_prediction_service as mps

        self.assertEqual(
            mps._KERAS_BACKEND,
            "tf_keras",
            "tf-keras is not installed - the model will fail to load",
        )

    def test_model_loads_and_predicts(self):
        import numpy as np

        from ml_model_serving.model_prediction_service import ModelPredictionService

        service = ModelPredictionService()
        size = 299
        dummy = np.zeros((1, size, size, 3), dtype="float32")

        prediction = service.predict_model(dummy)

        self.assertEqual(prediction.shape, (1, 1))
        self.assertGreaterEqual(float(prediction[0][0]), 0.0)
        self.assertLessEqual(float(prediction[0][0]), 1.0)

    def test_probability_helpers_are_complementary(self):
        from ml_model_serving.model_prediction_service import (
            melanoma_probability,
            not_melanoma_percent,
        )

        raw = 0.61  # model outputs P(NotMelanoma)
        self.assertAlmostEqual(melanoma_probability(raw), 0.39, places=6)
        self.assertAlmostEqual(not_melanoma_percent(raw), 61.0, places=6)


if __name__ == "__main__":
    unittest.main()
