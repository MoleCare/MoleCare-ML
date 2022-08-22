from ml_model_serving import app

from flask import request, jsonify, json, abort
from werkzeug.exceptions import HTTPException
from flask_cors import CORS, cross_origin
import os

from ml_model_serving.ImageProcessor import ImageProcessor
from ml_model_serving.ModelPredictionService import ModelPrediction
from ml_model_serving.Validator import Validator

# cross domain requests
CORS(app)

CLASSES = ['NotMelanoma', 'Melanoma']


@app.route('/')
@cross_origin()
def index():
    return "Working"


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    request_body = request.get_json()
    validator = Validator()
    prediction_id = request_body["predictionid"]
    image_base64 = request_body["imagebase64"]

    if validator.validate(prediction_id, image_base64) is False:
        abort(400)

    print("predictionId: ", prediction_id)

    imageProcessor = ImageProcessor()
    input_image = imageProcessor.prepare_input_image(image_base64)

    modelPrediction = ModelPrediction()
    prediction_res = modelPrediction.predict_model(input_image)

    prediction = prediction_res[0][0]
    # class_name = CLASSES[int(prediction > 0.5)]
    percentage = prediction * 100

    response_body = {}
    # response_body["prediction"] = class_name
    response_body["percent"] = percentage
    response_body["predictionid"] = prediction_id

    return jsonify({"status": 200, "data": response_body})


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
    # serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)
