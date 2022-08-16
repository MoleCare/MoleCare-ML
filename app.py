from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import os
import requests
import json
import numpy as np
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)
#cross domain requests
CORS(app)

XCEPTION_INPUT_SHAPE_SIZE = 299
URL_SERVING = 'http://tensorflow-serving:8501/v1/models/melanoma/1:predict'
CLASSES = ['Melanoma', 'NotMelanoma']

@app.route('/')
@cross_origin()
def index():
    return "Working"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    request_body = request.get_json()
    image_base64 = request_body['imagebase64']
    image_path = './test.jpeg'

    if os.path.exists(image_path):
        os.remove(image_path)

    img = Image.open(BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))
    img.save(image_path, 'jpeg')
    input_image = tf.keras.preprocessing.image.load_img(image_path)

    input_image = tf.image.resize(input_image,
                                  [XCEPTION_INPUT_SHAPE_SIZE, XCEPTION_INPUT_SHAPE_SIZE])
    input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    # this line is added because of a bug in tf_serving < 1.11
    input_image = input_image.astype('float16')

    payload = {
        "instances": [{"input_image": input_image.tolist()}]
    }

    print(URL_SERVING)
    response = requests.post(URL_SERVING, json=payload)

    result = json.loads(response.text)
    print("============")
    print(result)
    print("============")
    #prediction = result['predictions'][0]
    #class_name = CLASSES[int(prediction > 0.5)]
    #percentage = prediction * 100

    response_body = {}
    #response_body["prediction"] = class_name
    #response_body["percent"] = percentage

    return jsonify({"status": 200, "data": result})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)