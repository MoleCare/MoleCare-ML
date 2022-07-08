from flask import Flask, redirect, url_for, \
    request
from keras.preprocessing import image
from flask_cors import CORS, cross_origin
import os
import model
import requests
import json
import jsonify
import numpy as np
import re
from datetime import datetime

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
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + \
              "! It's " + formatted_now
    return content

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            class_name = model.get_prediction(image_path)
            result = {
                'class_name': class_name,
                'image_path': image_path,
            }
            return result
    return {'error': 'filename is empty string'}


#r = requests.post('http://image-serving:8501/v1/models/qualitynet:predict', json=payload)
# was "http://localhost:8501"
@app.route('/image-quality')
def image_quality():
    data = {}

    if not request.files["image"]:
        return jsonify({"status": 400, "message": 'No image passed'})

    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(request.files["image"].read()),
                                            target_size=(150, 150))) / 255.

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/qualitynet:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    pred = (np.array(pred['predictions'])[0] > 0.4).astype(np.int)
    if pred == 0:
        prediction = 'Bad'
    else:
        prediction = 'Good'

    data["prediction"] = prediction

    # Returning JSON response
    return jsonify({"status": 200, "message": 'No image passed', "data": data })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)