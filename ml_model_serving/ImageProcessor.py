import tensorflow as tf
import os
import numpy as np
from io import BytesIO
import base64
from PIL import Image


class ImageProcessor:

    XCEPTION_INPUT_SHAPE_SIZE = 299

    def prepare_input_image(image_base64):
        image_path = './test.jpeg'

        if os.path.exists(image_path):
            os.remove(image_path)

        with Image.open(BytesIO(base64.decodebytes(bytes(image_base64, "utf-8")))) as img:
            img.save(image_path, 'jpeg')

        input_image = tf.keras.preprocessing.image.load_img(image_path)

        input_image = tf.keras.preprocessing.image.img_to_array(input_image)
        input_image = tf.image.resize(input_image,
                                      [ImageProcessor.XCEPTION_INPUT_SHAPE_SIZE, ImageProcessor.XCEPTION_INPUT_SHAPE_SIZE])
        input_image = tf.keras.applications.xception.preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)

        # this line is added because of a bug in tf_serving < 1.11
        input_image = input_image.astype('float16')

        return input_image