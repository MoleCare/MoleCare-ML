import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util

channel = implementations.insecure_channel('localhost', 9000)

stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()

request.model_spec.name = '<the model name used above when starting TF serving>'
request.model_spec.signature_name = '<get this from Saved Model CLI output>'

request.inputs['<input name>'].CopyFrom(tf.make_tensor_proto(
                    '<a list of values, one or more e.g., '
                    '[1.0,2.0,3.0] corresponding to <input name>]>,'
                    ' dtype=<The appropriate dtype of the values e.g., '
                    'tf.float32, again the Saved Model CLI output will help>'
    )
)

# 60 is the timeout in seconds, but its blazing fast
result = stub.Predict(request, 60.0)

print(result)

#To get out a particular output as a numpy array, use:
tensor_util.MakeNdarray(result.outputs['<output name>'])



