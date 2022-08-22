import tensorflow as tf


class ModelPredictionService:
    def predict_model(self, input_image):
        model = tf.keras.models.load_model("./cnn-models/xception/1/")
        prediction = model.predict(input_image.tolist())
        print("prediction:", prediction)

        return prediction
