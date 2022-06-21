"""

import tensorflow as tf
import keras.applications
from src.preprocessing import preprocessing

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
def decode(image_bytes):
    #Takes a base64 encoded image and returns the preprocessed input tensor ready for the inference
# not working on GPU if tf.__version__ < 2.3, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_jpeg(contents=tf.io.decode_base64(x), channels=3))["output_0"],
            image_bytes,
            dtype=tf.float32,
        )
    return input_tensor

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
def decode_and_serve(image_bytes):
    return {
        tf.saved_model.CLASSIFY_OUTPUT_SCORES: classifier(decode(image_bytes=image_bytes)),
        tf.saved_model.CLASSIFY_OUTPUT_CLASSES: classifier.layers[1].columns,
    }

"""