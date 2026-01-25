import tensorflow as tf
import os
import logging
import time

logger = logging.getLogger(__name__)

# {'Melanoma': 0, 'NotMelanoma': 1}
# Values between 0 to 0.49 become class 0 (Melanoma)
# Values between 0.5 to 1 become class 1 (NotMelanoma)


class ModelPredictionService:
    """Singleton service for melanoma prediction using Xception model."""

    _instance = None
    _model = None
    _model_path = os.environ.get('MODEL_PATH', './cnn-models/xception/1/')

    def __new__(cls):
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
        prediction = self._model.predict(input_image.tolist(), verbose=0)
        inference_time = time.time() - start_time
        logger.debug(f"Inference completed in {inference_time*1000:.2f}ms")
        return prediction

    @classmethod
    def get_model(cls):
        """Get the loaded model instance."""
        if cls._model is None:
            cls._load_model()
        return cls._model
