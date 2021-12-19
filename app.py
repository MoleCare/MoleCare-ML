from flask import Flask, render_template, request, url_for, jsonify,Response
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


# Define a route for the default URL, which loads the form
@app.route('/inference', methods=['POST'])
def inference():
    request_data = request.json
    input_data = np.expand_dims(np.array(request_data), 0)
    result, label = run.inference(input_data)
    di={"result":str(result),'label': label[0].tolist()}
    return Response(json.dumps(di), mimetype='application/json')

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