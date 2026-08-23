import base64
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from PIL import Image

app = Flask(__name__)
#cross domain requests
CORS(app)

XCEPTION_INPUT_SHAPE_SIZE = 299
#URL_SERVING = 'http://tensorflow-serving:8501/v1/models/melanoma/1:predict'
URL_SERVING = 'http://localhost:8501/v1/models/melanoma/1:predict'
CLASSES = ['NotMelanoma', 'Melanoma']

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

    with Image.open(BytesIO(base64.decodebytes(bytes(image_base64, "utf-8")))) as img:
        img.save(image_path, 'jpeg')

    input_image = tf.keras.preprocessing.image.load_img(image_path)

    input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image = tf.image.resize(input_image,
                                  [XCEPTION_INPUT_SHAPE_SIZE,XCEPTION_INPUT_SHAPE_SIZE])
    input_image = tf.keras.applications.xception.preprocess_input(input_image)
    input_image = np.expand_dims(input_image, axis=0)

    #input_image = tf.keras.preprocessing.image.load_img(image_path)
    #input_image = np.array(input_image)
    #input_image = tf.image.resize(input_image,
    #                             [XCEPTION_INPUT_SHAPE_SIZE, XCEPTION_INPUT_SHAPE_SIZE])

    #input_image = np.expand_dims(input_image, axis=0)
    # this line is added because of a bug in tf_serving < 1.11
    input_image = input_image.astype('float16')

    model = tf.keras.models.load_model("./cnn-models/xception/1/")
    prediction_res = model.predict(input_image.tolist())
    print("prediction shape:", prediction_res)

    #encoded_input_string = base64.b64encode(input_image)
    #input_string = encoded_input_string.decode("utf-8")
    #instance = [{"b64": input_string}]
    #payload = json.dumps({"instances": instance})
    #response = requests.post(URL_SERVING, data=payload)

    #result = json.loads(response.text)
    print(prediction_res)
    prediction = prediction_res[0][0]
    class_name = CLASSES[int(prediction > 0.5)]
    percentage = prediction * 100

    response_body = {}
    response_body["prediction"] = class_name
    response_body["percent"] = percentage

    return jsonify({"status": 200, "data": response_body})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)
