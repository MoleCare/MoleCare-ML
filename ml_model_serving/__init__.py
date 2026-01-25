from flask import Flask
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Import controller
import ml_model_serving.PredictionController

# Warm up model on startup (if not in Lambda - Lambda handles this separately)
if os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is None:
    try:
        from ml_model_serving.ModelPredictionService import ModelPredictionService
        service = ModelPredictionService()
        ModelPredictionService.warmup()
        logger.info("Model initialized and warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
