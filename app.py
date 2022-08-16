from io import BytesIO

from flask import Flask, request
from keras.preprocessing import image
from flask_cors import CORS, cross_origin
import os
import xception_model
import requests
import json
import jsonify
import numpy as np
from datetime import datetime

app = Flask(__name__)

#cross domain requests
CORS(app)

@app.route('/')
@cross_origin()
def index():
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    content = "Hello, " + formatted_now
    return content

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    data = {}

    request_body = request.get_json()
    return request_body

    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(request.files["image"].read()),
                                            target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/default:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    percentage = np.array(pred['predictions'])[0] * 100

    pred = (np.array(pred['predictions'])[0] > 0.4).astype(np.int)
    if pred == 0:
        prediction = 'Bad'
    else:
        prediction = 'Good'

    data["prediction"] = prediction
    data["percent"] = percentage

    return jsonify({"status": 200, "data": data})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)