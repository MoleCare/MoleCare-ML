from ml_model_serving import app

from flask import request, jsonify, json, abort
from werkzeug.exceptions import HTTPException
from flask_cors import CORS, cross_origin
import os
import logging
import traceback
from ml_model_serving.ImageProcessor import ImageProcessor
from ml_model_serving.ModelPredictionService import ModelPredictionService
from ml_model_serving.Validator import Validator

# Import ABCDE analysis modules
try:
    from ml_model_serving.ABCDEAnalyzer import ABCDEAnalyzer
    from ml_model_serving.MoleAnalysisService import MoleAnalysisService
    ABCDE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ABCDE modules not available: {e}")
    ABCDE_AVAILABLE = False

# Import mole detection and evolution modules
try:
    from ml_model_serving.MoleDetectionService import MoleDetectionService
    from ml_model_serving.EvolutionAnalysisService import EvolutionAnalysisService
    DETECTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Detection modules not available: {e}")
    DETECTION_AVAILABLE = False

# Import Derm Foundation (Google Health AI) module
try:
    from ml_model_serving.DermFoundationService import DermFoundationService
    DERM_FOUNDATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Derm Foundation module not available: {e}")
    DERM_FOUNDATION_AVAILABLE = False

# cross domain requests
CORS(app)

# Initialize services
_mole_analysis_service = None
_mole_detection_service = None
_evolution_analysis_service = None
_derm_foundation_service = None

def get_mole_analysis_service():
    """Lazy initialization of MoleAnalysisService."""
    global _mole_analysis_service
    if _mole_analysis_service is None and ABCDE_AVAILABLE:
        _mole_analysis_service = MoleAnalysisService()
    return _mole_analysis_service

def get_mole_detection_service():
    """Lazy initialization of MoleDetectionService."""
    global _mole_detection_service
    if _mole_detection_service is None and DETECTION_AVAILABLE:
        _mole_detection_service = MoleDetectionService()
    return _mole_detection_service

def get_evolution_analysis_service():
    """Lazy initialization of EvolutionAnalysisService."""
    global _evolution_analysis_service
    if _evolution_analysis_service is None and DETECTION_AVAILABLE:
        _evolution_analysis_service = EvolutionAnalysisService()
    return _evolution_analysis_service

def get_derm_foundation_service():
    """Lazy initialization of DermFoundationService."""
    global _derm_foundation_service
    if _derm_foundation_service is None and DERM_FOUNDATION_AVAILABLE:
        _derm_foundation_service = DermFoundationService()
    return _derm_foundation_service


# Legal disclaimer included in all advanced analysis responses
MEDICAL_DISCLAIMER = (
    'This analysis is for informational purposes only and does not constitute '
    'medical advice or diagnosis. This tool uses Google Health AI Developer '
    'Foundations under their Terms of Use. Always consult a qualified '
    'healthcare professional for medical concerns.'
)


@app.route('/')
@cross_origin()
def index():
    return "Working"


@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Comprehensive health check for all ML services."""
    import time as time_module
    start_time = time_module.time()

    health = {
        'status': 'healthy',
        'services': {}
    }

    # Check Xception model
    try:
        model = ModelPredictionService()
        health['services']['xception'] = {'status': 'up', 'type': 'baseline'}
    except Exception as e:
        health['services']['xception'] = {'status': 'down', 'error': str(e)}
        health['status'] = 'degraded'

    # Check ABCDE analyzer
    health['services']['abcde'] = {
        'status': 'up' if ABCDE_AVAILABLE else 'unavailable',
        'type': 'analysis'
    }

    # Check Detection service
    health['services']['detection'] = {
        'status': 'up' if DETECTION_AVAILABLE else 'unavailable',
        'type': 'detection'
    }

    # Check Derm Foundation (premium)
    if DERM_FOUNDATION_AVAILABLE:
        try:
            derm = get_derm_foundation_service()
            if derm and derm.is_available():
                health['services']['derm_foundation'] = {
                    'status': 'up',
                    'type': 'advanced',
                    'premium': True
                }
            else:
                health['services']['derm_foundation'] = {
                    'status': 'loading',
                    'type': 'advanced',
                    'premium': True
                }
        except Exception as e:
            health['services']['derm_foundation'] = {
                'status': 'down',
                'error': str(e),
                'premium': True
            }
    else:
        health['services']['derm_foundation'] = {
            'status': 'unavailable',
            'type': 'advanced',
            'premium': True,
            'note': 'Install huggingface_hub and configure access to enable'
        }

    # If Xception is down, the whole service is unhealthy
    if health['services']['xception'].get('status') == 'down':
        health['status'] = 'unhealthy'

    health['response_time_ms'] = int((time_module.time() - start_time) * 1000)

    status_code = 200 if health['status'] != 'unhealthy' else 503
    return jsonify(health), status_code


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    request_body = request.get_json()
    validator = Validator()
    prediction_id = request_body["predictionid"]
    image_base64 = request_body["imagebase64"]

    if validator.validate(prediction_id, image_base64) is False:
        abort(400)

    app.logger.info('predictionId: %s', prediction_id)

    image_processor = ImageProcessor()
    input_image = image_processor.prepare_input_image(image_base64)
    model_prediction = ModelPredictionService()
    prediction_res = model_prediction.predict_model(input_image)
    app.logger.info('prediction: %s', prediction_res)
    prediction_value = prediction_res[0][0]
    prediction_percent = prediction_value * 100

    response_body = {}
    response_body["percent"] = prediction_percent
    response_body["predictionid"] = prediction_id

    return jsonify({"status": 200, "data": response_body})


@app.route('/analyze', methods=['POST'])
@cross_origin()
def analyze():
    """Combined ML prediction + ABCDE analysis endpoint."""
    if not ABCDE_AVAILABLE:
        abort(503, description="ABCDE analysis module not available")

    try:
        request_body = request.get_json()
        validator = Validator()

        prediction_id = request_body.get("predictionid")
        image_base64 = request_body.get("imagebase64")
        reference_mm = request_body.get("reference_mm")  # Optional calibration

        if validator.validate(prediction_id, image_base64) is False:
            abort(400, description="Invalid prediction ID or image data")

        app.logger.info('Comprehensive analysis for predictionId: %s', prediction_id)

        # Get analysis service
        mole_service = get_mole_analysis_service()
        if mole_service is None:
            abort(503, description="Mole analysis service not initialized")

        # Perform comprehensive analysis
        result = mole_service.analyze_mole(image_base64, reference_mm=reference_mm)
        response_data = mole_service.to_api_response(result)
        response_data["predictionid"] = prediction_id

        app.logger.info('Analysis complete - Risk: %s, Urgent: %s',
                       result.combined_risk_level, result.urgent_referral)

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Analysis error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Analysis failed: {str(e)}")


@app.route('/analyze/abcde', methods=['POST'])
@cross_origin()
def analyze_abcde_only():
    """ABCDE-only analysis endpoint (without ML prediction)."""
    if not ABCDE_AVAILABLE:
        abort(503, description="ABCDE analysis module not available")

    try:
        request_body = request.get_json()

        image_base64 = request_body.get("imagebase64")
        reference_mm = request_body.get("reference_mm")

        if not image_base64:
            abort(400, description="Missing imagebase64 field")

        app.logger.info('ABCDE-only analysis requested')

        # Decode image
        import base64
        import cv2
        import numpy as np

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            abort(400, description="Invalid image data")

        # Perform ABCDE analysis
        analyzer = ABCDEAnalyzer()
        abcde_score = analyzer.analyze(image, reference_mm=reference_mm)

        response_data = {
            "asymmetry_score": abcde_score.asymmetry_score,
            "border_score": abcde_score.border_score,
            "color_score": abcde_score.color_score,
            "diameter_mm": abcde_score.diameter_mm,
            "total_score": abcde_score.total_score,
            "risk_level": abcde_score.risk_level,
            "findings": abcde_score.findings
        }

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('ABCDE analysis error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"ABCDE analysis failed: {str(e)}")


@app.route('/compare', methods=['POST'])
@cross_origin()
def compare_moles():
    """Compare two mole images for evolution tracking."""
    if not ABCDE_AVAILABLE:
        abort(503, description="ABCDE analysis module not available")

    try:
        request_body = request.get_json()

        image1_base64 = request_body.get("image1_base64")
        image2_base64 = request_body.get("image2_base64")
        date1 = request_body.get("date1")  # Optional: date of first image
        date2 = request_body.get("date2")  # Optional: date of second image

        if not image1_base64 or not image2_base64:
            abort(400, description="Both image1_base64 and image2_base64 are required")

        app.logger.info('Mole comparison requested')

        # Get analysis service
        mole_service = get_mole_analysis_service()
        if mole_service is None:
            abort(503, description="Mole analysis service not initialized")

        # Perform comparison
        comparison_result = mole_service.compare_moles(image1_base64, image2_base64)

        response_data = {
            "evolution_score": comparison_result.get("evolution_score", 0),
            "size_change_percent": comparison_result.get("size_change_percent", 0),
            "color_change_score": comparison_result.get("color_change_score", 0),
            "shape_change_score": comparison_result.get("shape_change_score", 0),
            "significant_changes": comparison_result.get("significant_changes", []),
            "recommendations": comparison_result.get("recommendations", []),
            "requires_attention": comparison_result.get("requires_attention", False)
        }

        if date1 and date2:
            response_data["date1"] = date1
            response_data["date2"] = date2

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Comparison error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Comparison failed: {str(e)}")


@app.route('/detect', methods=['POST'])
@cross_origin()
def detect_moles():
    """Detect moles in an image and return bounding boxes with metadata."""
    if not DETECTION_AVAILABLE:
        abort(503, description="Mole detection module not available")

    try:
        request_body = request.get_json()

        image_base64 = request_body.get("imagebase64")
        return_mask = request_body.get("return_mask", False)
        return_cropped = request_body.get("return_cropped", True)
        crop_padding = request_body.get("crop_padding", 20)

        if not image_base64:
            abort(400, description="Missing imagebase64 field")

        app.logger.info('Mole detection requested')

        # Get detection service
        detection_service = get_mole_detection_service()
        if detection_service is None:
            abort(503, description="Mole detection service not initialized")

        # Perform detection
        result = detection_service.detect_moles(
            image_base64,
            return_mask=return_mask,
            return_cropped=return_cropped,
            crop_padding=crop_padding
        )

        # Convert to API response
        moles_data = []
        for mole in result.detected_moles:
            mole_data = {
                "id": mole.id,
                "bounding_box": {
                    "x": mole.bounding_box[0],
                    "y": mole.bounding_box[1],
                    "width": mole.bounding_box[2],
                    "height": mole.bounding_box[3]
                },
                "center": {"x": mole.center[0], "y": mole.center[1]},
                "area_pixels": mole.area_pixels,
                "confidence": mole.confidence,
                "is_primary": mole.is_primary
            }
            if mole.cropped_image_base64:
                mole_data["cropped_image"] = mole.cropped_image_base64
            moles_data.append(mole_data)

        response_data = {
            "moles_detected": result.moles_detected,
            "processing_time_ms": result.processing_time_ms,
            "image_dimensions": {
                "width": result.image_width,
                "height": result.image_height
            },
            "moles": moles_data
        }

        if result.full_mask_base64:
            response_data["mask_image"] = result.full_mask_base64

        app.logger.info(f'Detection complete - {result.moles_detected} moles found')

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Detection error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Detection failed: {str(e)}")


@app.route('/detect/extract', methods=['POST'])
@cross_origin()
def extract_mole():
    """Extract and enhance a specific mole from an image."""
    if not DETECTION_AVAILABLE:
        abort(503, description="Mole detection module not available")

    try:
        request_body = request.get_json()

        image_base64 = request_body.get("imagebase64")
        mole_id = request_body.get("mole_id")  # Optional: specific mole to extract

        if not image_base64:
            abort(400, description="Missing imagebase64 field")

        app.logger.info(f'Mole extraction requested, mole_id={mole_id}')

        # Get detection service
        detection_service = get_mole_detection_service()
        if detection_service is None:
            abort(503, description="Mole detection service not initialized")

        # Perform extraction
        result = detection_service.extract_and_enhance_mole(image_base64, mole_id=mole_id)

        if result.moles_detected == 0:
            return jsonify({
                "status": 200,
                "data": {
                    "moles_detected": 0,
                    "message": "No moles detected in the image"
                }
            })

        # Get the primary mole
        primary_mole = next((m for m in result.detected_moles if m.is_primary), result.detected_moles[0])

        response_data = {
            "moles_detected": result.moles_detected,
            "processing_time_ms": result.processing_time_ms,
            "extracted_mole": {
                "id": primary_mole.id,
                "bounding_box": {
                    "x": primary_mole.bounding_box[0],
                    "y": primary_mole.bounding_box[1],
                    "width": primary_mole.bounding_box[2],
                    "height": primary_mole.bounding_box[3]
                },
                "center": {"x": primary_mole.center[0], "y": primary_mole.center[1]},
                "area_pixels": primary_mole.area_pixels,
                "confidence": primary_mole.confidence,
                "enhanced_image": primary_mole.cropped_image_base64
            }
        }

        app.logger.info('Extraction complete')

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Extraction error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Extraction failed: {str(e)}")


@app.route('/evolution', methods=['POST'])
@cross_origin()
def analyze_evolution():
    """Analyze evolution of a mole across multiple timestamped images."""
    if not DETECTION_AVAILABLE:
        abort(503, description="Evolution analysis module not available")

    try:
        request_body = request.get_json()

        images = request_body.get("images")  # List of {imagebase64, timestamp, ?label}
        mole_id = request_body.get("mole_id")  # Optional: mole identifier
        reference_mm = request_body.get("reference_mm")  # Optional: calibration

        if not images or len(images) < 2:
            abort(400, description="At least 2 images with timestamps are required")

        # Validate each image has required fields
        for i, img in enumerate(images):
            if "imagebase64" not in img:
                abort(400, description=f"Image {i} missing imagebase64 field")
            if "timestamp" not in img:
                abort(400, description=f"Image {i} missing timestamp field")

        app.logger.info(f'Evolution analysis requested for {len(images)} images')

        # Get evolution service
        evolution_service = get_evolution_analysis_service()
        if evolution_service is None:
            abort(503, description="Evolution analysis service not initialized")

        # Perform evolution analysis
        report = evolution_service.analyze_evolution(
            images,
            mole_id=mole_id,
            reference_mm=reference_mm
        )

        # Convert snapshots to API format
        snapshots_data = []
        for snapshot in report.snapshots:
            snapshots_data.append({
                "timestamp": snapshot.timestamp,
                "label": snapshot.label,
                "asymmetry_score": snapshot.asymmetry_score,
                "border_score": snapshot.border_score,
                "color_score": snapshot.color_score,
                "diameter_mm": snapshot.diameter_mm,
                "total_abcde_score": snapshot.total_abcde_score,
                "image_quality": snapshot.image_quality
            })

        # Convert changes to API format
        changes_data = []
        for change in report.changes:
            changes_data.append({
                "from_timestamp": change.from_timestamp,
                "to_timestamp": change.to_timestamp,
                "days_elapsed": change.days_elapsed,
                "criteria": change.criteria,
                "change_value": change.change_value,
                "change_percent": change.change_percent,
                "severity": change.severity,
                "description": change.description
            })

        response_data = {
            "mole_id": report.mole_id,
            "analysis_timestamp": report.analysis_timestamp,
            "total_images_analyzed": report.total_images_analyzed,
            "time_span_days": report.time_span_days,
            "overall_risk_level": report.overall_risk_level,
            "evolution_detected": report.evolution_detected,
            "risk_trajectory": report.risk_trajectory,
            "snapshots": snapshots_data,
            "significant_changes": changes_data,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "urgent_referral": report.urgent_referral
        }

        app.logger.info(f'Evolution analysis complete - Risk: {report.overall_risk_level}, '
                       f'Trajectory: {report.risk_trajectory}')

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Evolution analysis error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Evolution analysis failed: {str(e)}")


@app.route('/validate', methods=['POST'])
@cross_origin()
def validate_image():
    """Validate image quality for mole analysis."""
    try:
        request_body = request.get_json()

        image_base64 = request_body.get("imagebase64")

        if not image_base64:
            abort(400, description="Missing imagebase64 field")

        app.logger.info('Image validation requested')

        # Validate image
        image_processor = ImageProcessor()
        quality_report = image_processor.validate_image_quality(image_base64)

        response_data = {
            "is_valid": quality_report.is_valid,
            "original_dimensions": {
                "width": quality_report.original_width,
                "height": quality_report.original_height
            },
            "aspect_ratio": quality_report.aspect_ratio,
            "format_detected": quality_report.format_detected,
            "file_size_kb": quality_report.file_size_kb,
            "meets_minimum_resolution": quality_report.meets_minimum_resolution,
            "is_optimal_resolution": quality_report.is_optimal_resolution,
            "warnings": quality_report.warnings,
            "errors": quality_report.errors,
            "recommendations": {
                "minimum_resolution": f"{ImageProcessor.MIN_RESOLUTION}x{ImageProcessor.MIN_RESOLUTION}",
                "optimal_resolution": f"{ImageProcessor.OPTIMAL_RESOLUTION}x{ImageProcessor.OPTIMAL_RESOLUTION}",
                "recommended_format": "JPEG",
                "recommended_quality": "85-95%"
            }
        }

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Validation error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Validation failed: {str(e)}")


@app.route('/predict-advanced', methods=['POST'])
@cross_origin()
def predict_advanced():
    """Advanced melanoma prediction using Google Derm Foundation model.

    This is a premium feature that uses dermatology-specific embeddings
    for improved accuracy over the baseline Xception model.
    """
    if not DERM_FOUNDATION_AVAILABLE:
        abort(503, description="Derm Foundation model not available")

    try:
        request_body = request.get_json()
        validator = Validator()

        prediction_id = request_body.get("predictionid")
        image_base64 = request_body.get("imagebase64")
        threshold = request_body.get("threshold", 0.5)

        if validator.validate(prediction_id, image_base64) is False:
            abort(400, description="Invalid prediction ID or image data")

        app.logger.info('Advanced prediction (Derm Foundation) for predictionId: %s', prediction_id)

        # Get Derm Foundation service
        derm_service = get_derm_foundation_service()
        if derm_service is None or not derm_service.is_available():
            abort(503, description="Derm Foundation service not initialized")

        # Decode image
        import base64
        image_bytes = base64.b64decode(image_base64)

        # Make advanced prediction
        result = derm_service.predict(image_bytes, threshold=threshold)

        if not result.get('success'):
            abort(500, description=result.get('error', 'Advanced prediction failed'))

        response_data = {
            "predictionid": prediction_id,
            "model_type": "derm_foundation",
            "melanoma_probability": result.get('melanoma_probability', 0),
            "prediction": result.get('prediction', 'Unknown'),
            "confidence": result.get('confidence', 0),
            "processing_time_ms": result.get('processing_time_ms', 0),
            "disclaimer": MEDICAL_DISCLAIMER
        }

        app.logger.info('Advanced prediction complete: %s (prob=%.3f)',
                       result.get('prediction'), result.get('melanoma_probability', 0))

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Advanced prediction error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Advanced prediction failed: {str(e)}")


@app.route('/compare-models', methods=['POST'])
@cross_origin()
def compare_models():
    """A/B comparison between Xception (baseline) and Derm Foundation (advanced).

    Runs both models and returns a combined analysis with weighted confidence.
    Premium feature.
    """
    if not DERM_FOUNDATION_AVAILABLE:
        abort(503, description="Derm Foundation model not available for comparison")

    try:
        request_body = request.get_json()
        validator = Validator()

        prediction_id = request_body.get("predictionid")
        image_base64 = request_body.get("imagebase64")
        threshold = request_body.get("threshold", 0.5)

        if validator.validate(prediction_id, image_base64) is False:
            abort(400, description="Invalid prediction ID or image data")

        app.logger.info('Model comparison for predictionId: %s', prediction_id)

        import base64
        import time as time_module

        start_time = time_module.time()

        # 1. Run baseline Xception prediction
        image_processor = ImageProcessor()
        input_image = image_processor.prepare_input_image(image_base64)
        model_prediction = ModelPredictionService()
        xception_result = model_prediction.predict_model(input_image)
        xception_value = float(xception_result[0][0])

        # Xception: 0 = Melanoma, 1 = NotMelanoma
        xception_melanoma_prob = 1 - xception_value
        xception_pred = 'Melanoma' if xception_melanoma_prob >= threshold else 'NotMelanoma'

        baseline_prediction = {
            'melanomaProbability': xception_melanoma_prob,
            'predictionResult': xception_pred
        }

        # 2. Run Derm Foundation prediction
        derm_service = get_derm_foundation_service()
        if derm_service is None or not derm_service.is_available():
            abort(503, description="Derm Foundation service not initialized")

        image_bytes = base64.b64decode(image_base64)
        comparison = derm_service.compare_with_baseline(
            image_bytes,
            baseline_prediction,
            threshold=threshold
        )

        total_time = time_module.time() - start_time

        if not comparison.get('success'):
            abort(500, description=comparison.get('error', 'Comparison failed'))

        response_data = {
            "predictionid": prediction_id,
            "baseline": comparison['baseline'],
            "advanced": comparison['advanced'],
            "combined": comparison['combined'],
            "analysis": comparison['analysis'],
            "total_processing_time_ms": int(total_time * 1000),
            "disclaimer": MEDICAL_DISCLAIMER
        }

        app.logger.info('Model comparison complete - Baseline: %s (%.3f), Advanced: %s (%.3f), Combined: %s',
                       comparison['baseline']['prediction'],
                       comparison['baseline']['melanoma_probability'],
                       comparison['advanced']['prediction'],
                       comparison['advanced']['melanoma_probability'],
                       comparison['combined']['prediction'])

        return jsonify({"status": 200, "data": response_data})

    except Exception as e:
        app.logger.error('Comparison error: %s\n%s', str(e), traceback.format_exc())
        abort(500, description=f"Model comparison failed: {str(e)}")


@app.route('/model-status', methods=['GET'])
@cross_origin()
def model_status():
    """Check availability of all ML models."""
    derm_service = get_derm_foundation_service() if DERM_FOUNDATION_AVAILABLE else None

    status = {
        "xception": {
            "available": True,
            "model_type": "baseline"
        },
        "derm_foundation": {
            "available": derm_service.is_available() if derm_service else False,
            "model_type": "advanced",
            "requires_premium": True
        },
        "abcde_analyzer": {
            "available": ABCDE_AVAILABLE,
            "model_type": "analysis"
        },
        "mole_detection": {
            "available": DETECTION_AVAILABLE,
            "model_type": "detection"
        }
    }

    return jsonify({"status": 200, "data": status})


@app.errorhandler(HTTPException)
def handle_exception(e):
    response = e.get_response()
    app.logger.error(e)

    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })

    response.content_type = "application/json"
    return response


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
