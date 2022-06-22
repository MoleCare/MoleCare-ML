from flask import Flask,\
    request, jsonify, \
    redirect, url_for
from flask_cors import CORS, cross_origin
from numpy import array
import os
import re
from datetime import datetime
import traceback

class_predictions = array(['Melanoma, NotMelanoma'])

app = Flask(__name__)
#cross domain requests
CORS(app)

@app.route('/')
@cross_origin()
def index():
    return redirect(url_for('home'))

@app.route('/hello/', methods=['GET', 'POST'])
@cross_origin()
def home():
    return "Hello"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    serve(app, host="0.0.0.0", port=port)


'''
import requests
import base64
from io import BytesIO
import tensorflow as tf
from keras.applications import inception_v3
from keras.preprocessing import image
import json
import numpy as np

@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
                                            target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])

def preprocess(self, image):
    image = tf.image.resize(image, (self.image_size, self.image_size))
    return tf.cast(image, tf.float32) / 255.0



@app.route('/prediction/', methods=['POST'])
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin = image_link
    )
    img = load_img(
        img_path,
        target_size = (224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)
    model_score = dumps(model_score.tolist())

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }
'''