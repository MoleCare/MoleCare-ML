from ml_model_serving import app

from flask import Flask, request, jsonify, json, abort
from werkzeug.exceptions import HTTPException
from flask_cors import CORS, cross_origin
import tensorflow as tf
import os
import numpy as np
from io import BytesIO
import base64
from PIL import Image

#cross domain requests
CORS(app)

XCEPTION_INPUT_SHAPE_SIZE = 299
CLASSES = ['NotMelanoma', 'Melanoma']

@app.route('/')
@cross_origin()
def index():
    return "Working"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    request_body = request.get_json()

    prediction_id = request_body["predictionid"]
    validate_prediction_id(prediction_id)
    print("predictionId: ", prediction_id)
    image_base64 = request_body["imagebase64"]
    validate_image_base64(image_base64)

    input_image = prepare_input_image(image_base64)
    prediction_res = predict_model(input_image)

    prediction = prediction_res[0][0]
    #class_name = CLASSES[int(prediction > 0.5)]
    percentage = prediction * 100

    response_body = {}
    #response_body["prediction"] = class_name
    response_body["percent"] = percentage
    response_body["predictionid"] = prediction_id

    return jsonify({"status": 200, "data": response_body})

def validate_prediction_id(prediction_id):
    if prediction_id is None or prediction_id == "":
        abort(400)

def validate_image_base64(image_base64):
    if image_base64 is None or image_base64 == "":
        abort(400)

def prepare_input_image(image_base64):
    image_path = './test.jpeg'

    if os.path.exists(image_path):
        os.remove(image_path)

    with Image.open(BytesIO(base64.decodebytes(bytes(image_base64, "utf-8")))) as img:
        img.save(image_path, 'jpeg')

    input_image = tf.keras.preprocessing.image.load_img(image_path)

    input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image = tf.image.resize(input_image,
                                  [XCEPTION_INPUT_SHAPE_SIZE,XCEPTION_INPUT_SHAPE_SIZE])
    input_image = tf.keras.applications.xception.preprocess_input(input_image)
    input_image = np.expand_dims(input_image, axis=0)

    # this line is added because of a bug in tf_serving < 1.11
    input_image = input_image.astype('float16')

    return input_image

def predict_model(input_image):
    model = tf.keras.models.load_model("./cnn-models/xception/1/")
    prediction_res = model.predict(input_image.tolist())
    print("prediction shape:", prediction_res)

    return prediction_res

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
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)