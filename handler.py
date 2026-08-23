"""
AWS Lambda handler for melanoma prediction.

This handler is designed to work with Lambda Container images.
It initializes the model once (singleton) and handles prediction requests.
"""
import base64
import json
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model at module level (cold start optimization)
model_service = None
image_processor = None


def _initialize():
    """Initialize model and image processor (called once per container)."""
    global model_service, image_processor

    if model_service is None:
        logger.info("Initializing model service...")
        from ml_model_serving.image_processor import ImageProcessor
        from ml_model_serving.model_prediction_service import ModelPredictionService

        model_service = ModelPredictionService()
        image_processor = ImageProcessor()

        # Warm up the model
        ModelPredictionService.warmup()
        logger.info("Model initialization complete")


# Initialize on import (Lambda keeps container warm)
_initialize()


def predict(event, context):
    """
    Lambda handler for melanoma prediction.

    Expected request body (JSON):
    {
        "predictionid": "uuid-string",
        "imagebase64": "base64-encoded-image"
    }

    Returns:
    {
        "statusCode": 200,
        "body": {
            "status": 200,
            "data": {
                "melanomaProbability": float,   # P(melanoma), 0-1
                "percent": float,                # DEPRECATED: P(NOT melanoma), 0-100
                "predictionid": "uuid-string"
            }
        }
    }
    """
    start_time = time.time()

    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        prediction_id = body.get('predictionid')
        image_base64 = body.get('imagebase64')

        # Validate inputs
        if not prediction_id or not image_base64:
            return _error_response(400, "Missing required fields: predictionid and imagebase64")

        logger.info(f"Processing prediction request: {prediction_id}")

        # Process image and predict
        input_image = image_processor.prepare_input_image(image_base64)
        prediction_res = model_service.predict_model(input_image)

        from ml_model_serving.model_prediction_service import (
            melanoma_probability,
            not_melanoma_percent,
        )

        raw_score = prediction_res[0][0]
        # P(melanoma) -- same polarity as every other endpoint.
        melanoma_prob = melanoma_probability(raw_score)
        # DEPRECATED: P(NotMelanoma) as 0-100, so HIGH means LOW risk.
        prediction_percent = not_melanoma_percent(raw_score)

        # Calculate metrics
        inference_time = (time.time() - start_time) * 1000  # ms

        response_data = {
            "melanomaProbability": melanoma_prob,
            "percent": prediction_percent,
            "predictionid": prediction_id,
            "inference_time_ms": round(inference_time, 2)
        }

        logger.info(f"Prediction complete: {prediction_id}, melanoma_prob={melanoma_prob:.4f}, time={inference_time:.2f}ms")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
            },
            "body": json.dumps({
                "status": 200,
                "data": response_data
            })
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return _error_response(400, str(e))

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return _error_response(500, "Internal server error")


def _error_response(status_code, message):
    """Create error response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "status": status_code,
            "error": message
        })
    }


def health(event, context):
    """Health check endpoint for monitoring."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "healthy",
            "model_loaded": model_service is not None
        })
    }
