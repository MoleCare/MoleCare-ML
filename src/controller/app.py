from flask import Flask, render_template, \
    request, url_for, jsonify, \
    Response, jsonify, make_response, \
    abort, redirect, url_for
from paramiko._winapi import get_current_user
from werkzeug.utils import secure_filename
from markupsafe import escape
import requests
import base64
from io import BytesIO
import tensorflow as tf
from keras.applications import inception_v3
from keras.preprocessing import image
import json
import numpy as np
import re
import os
import argparse
import sys
from datetime import datetime

app = Flask(__name__)

# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.errorhandler(404)
def not_found(error):
    resp = make_response(render_template('error.html'), 404)
    resp.headers['X-Something'] = 'A value'
    return resp

@app.route("/me")
def me_api():
    user = get_current_user()
    return {
        "username": user.username,
        "theme": user.theme,
        "image": url_for("user_image", filename=user.image),
    }

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['the_file']
        file.save(f"/var/www/uploads/{secure_filename(file.filename)}")

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {escape(username)}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return f'Subpath {escape(subpath)}'

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


@app.route('/hello/', methods=['GET', 'POST'])
def home():
    return "Hello"

#@app.route(APP_ROOT, methods=["POST"])
#def infer():
#    data = request.json
#    image = data['image']
#    return u_net.infer(image)


#@app.before_request
#def load_user():
#    if "user_id" in session:
#        g.user = db.session.get(session["user_id"])

#@app.route("/me")
#def me_api():
#    user = get_current_user()
#    return {
#        "username": user.username,
#        "theme": user.theme,
#        "image": url_for("user_image", filename=user.image),
#    }

#@app.route("/users")
#def users_api():
#    users = get_all_users()
#    return jsonify([user.to_json() for user in users])

# Define a route for the default URL, which loads the form
@app.route('/inference', methods=['POST'])
def inference():
    request_data = request.json
    input_data = np.expand_dims(np.array(request_data), 0)
    result, label = '' #run.inference(input_data)
    di={"result":str(result),'label': label[0].tolist()}
    return Response(json.dumps(di), mimetype='application/json')

def preprocess(self, image):
    image = tf.image.resize(image, (self.image_size, self.image_size))
    return tf.cast(image, tf.float32) / 255.0

def infer(self, image=None):
    tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor_image = self.preprocess(tensor_image)
    shape = tensor_image.shape
    tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
    return self.predict(tensor_image)['conv2d_transpose_4']



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

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.txt')


#Coolie
#@app.route('/')
#def index():
#    username = request.cookies.get('username')
    # use cookies.get(key) instead of cookies[key] to not get a
    # KeyError if the cookie is missing.
#
# from flask import make_response
#@app.route('/')
#def index():
#    resp = make_response(render_template(...))
#    resp.set_cookie('username', 'the username')
#    return resp


#from flask import abort, redirect, url_for
#@app.route('/')
#def index():
#    return redirect(url_for('login'))
#
#@app.route('/login')
#def login():
#    abort(401)
#    this_is_never_executed()


#error handler
#from flask import render_template
#@app.errorhandler(404)
#def page_not_found(error):
#    return render_template('page_not_found.html'), 404
#
#@app.errorhandler(Exception)
#def handle_exception(e):
#    return jsonify(stackTrace=traceback.format_exc())









