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