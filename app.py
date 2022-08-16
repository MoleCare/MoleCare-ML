from flask import Flask, request
from flask_cors import CORS, cross_origin
import jsonify
import tensorflow as tf
import os
import requests
import json
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)
#cross domain requests
CORS(app)

XCEPTION_INPUT_SHAPE_SIZE = 299
MODEL_URI = 'http://localhost:8501/v1/models/default:predict'
CLASSES = ['Melanoma', 'NotMelanoma']

@app.route('/')
@cross_origin()
def index():
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    content = "Hello, " + formatted_now
    return content

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = {}

    request_body = request.get_json()
    imagebase64 = request_body['imagebase64']

    print(imagebase64)

    imagePath = './test.jpeg'
    img = Image.open(BytesIO(base64.decodebytes(bytes(imagebase64, "utf-8"))))
    img.save(imagePath, 'jpeg')

    input_image = tf.keras.preprocessing.image.load_img(imagePath)

    os.remove(imagePath)

    input_image = tf.image.resize(input_image,
                                  [XCEPTION_INPUT_SHAPE_SIZE, XCEPTION_INPUT_SHAPE_SIZE])
    input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    #input_image = tf.keras.applications.xception.preprocess_input(input_image)
    input_image = np.expand_dims(input_image, axis=0)

    # this line is added because of a bug in tf_serving < 1.11
    input_image = input_image.astype('float16')

    payload = {
        "instances": [{'input_image': input_image.tolist()}]
    }

    print(payload)

    response = requests.post('http://localhost:8502/v1/models/xception:predict', json = payload)

    result = json.loads(response.text)
    prediction = np.squeeze(result['predictions'][0])
    class_name = CLASSES[int(prediction > 0.5)]
    percentage = prediction * 100

    data["prediction"] = class_name
    data["percent"] = percentage

    return jsonify({"status": 200, "data": data})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)