import tensorflow as tf
#from pymemcache.client import base

#cache = base.Client(('localhost', 11211))

#{'Melanoma': 0, 'NotMelanoma': 1}
#Values between 0 to 0.49 become class 0 (Melanoma)
#Values between 0.5 to 1 become class 1 (NotMelanoma)

class ModelPredictionService:

    def predict_model(self, input_image):

    #    prediction_model = cache.get('model')

    #    if prediction_model is None:
    #        prediction_model = tf.keras.models.load_model("./cnn-models/xception/1/")
    #        cache.set('model', prediction_model, timeout=0)  # 0 - never expires

        prediction_model = tf.keras.models.load_model("./cnn-models/xception/1/")
        prediction = prediction_model.predict(input_image.tolist())

        return prediction
