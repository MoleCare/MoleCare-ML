import logging
import os
import threading
import time

import tensorflow as tf

logger = logging.getLogger(__name__)

# The model was trained with {'Melanoma': 0, 'NotMelanoma': 1}, so the raw score
# is P(NotMelanoma): a value near 1 means the lesion is probably NOT melanoma.
#
# That polarity is the opposite of what almost every caller expects, and reading
# `prediction[0][0]` directly is exactly how /predict and /predict-advanced came
# to report opposite things about the same image. Go through the helpers below.


def melanoma_probability(raw_score) -> float:
    """P(melanoma) in 0..1, from the model's raw P(NotMelanoma) output."""
    return 1.0 - float(raw_score)


def not_melanoma_percent(raw_score) -> float:
    """The legacy `percent` field: P(NotMelanoma) as 0..100.

    Kept only for the existing molecare-server consumer. Prefer
    `melanoma_probability`; a high `percent` means LOW melanoma risk.
    """
    return float(raw_score) * 100


class ModelPredictionService:
    """Singleton service for melanoma prediction using Xception model."""

    _instance = None
    _model = None
    _model_path = os.environ.get('MODEL_PATH', './cnn-models/xception/1/')
    # Without this, two requests arriving during a cold start on a threaded
    # worker can both pass the `is None` check and load the model twice.
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelPredictionService, cls).__new__(cls)
                    cls._load_model()
        return cls._instance

    @classmethod
    def _load_model(cls):
        """Load the TensorFlow model once at startup."""
        if cls._model is None:
            logger.info(f"Loading model from: {cls._model_path}")
            start_time = time.time()
            cls._model = tf.keras.models.load_model(cls._model_path)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

    @classmethod
    def warmup(cls):
        """Warm up the model with a dummy prediction to reduce cold start latency."""
        import numpy as np
        logger.info("Warming up model...")
        dummy_input = np.zeros((1, 299, 299, 3), dtype='float16')
        cls._model.predict(dummy_input, verbose=0)
        logger.info("Model warmup complete")

    def predict_model(self, input_image):
        """Make a prediction using the loaded model."""
        start_time = time.time()
        # Pass the ndarray straight through. `.tolist()` here cost ~15ms per
        # request converting 268,203 values into Python floats, which Keras
        # then had to re-parse back into a tensor.
        prediction = self._model.predict(input_image, verbose=0)
        inference_time = time.time() - start_time
        logger.debug(f"Inference completed in {inference_time*1000:.2f}ms")
        return prediction

    @classmethod
    def get_model(cls):
        """Get the loaded model instance."""
        if cls._model is None:
            cls._load_model()
        return cls._model
