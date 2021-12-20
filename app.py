from flask import Flask, render_template, request, url_for, jsonify,Response, jsonify
import json
import tensorflow as tf
import numpy as np
import re
import os
import argparse
import sys
from datetime import datetime

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)


class mainSessRunning():
    def __init__(self):
        host, port = FLAGS.server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'example_model'
        self.request.model_spec.signature_name = 'prediction'

    def inference(self, val_x):
        # temp_data = numpy.random.randn(100, 3).astype(numpy.float32)
        temp_data = val_x.astype(np.float32).reshape(-1, 3)
        print("temp_data is:", temp_data)
        data, label = temp_data, np.sum(temp_data * np.array([1, 2, 3]).astype(np.float32), 1)
        self.request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=data.shape))

        result = self.stub.Predict(self.request, 5.0)
        return result, label


run = mainSessRunning()

print("Initialization done. ")

@app.route("/")
def home():
    return "Hello, Flask!"

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
    result, label = run.inference(input_data)
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









