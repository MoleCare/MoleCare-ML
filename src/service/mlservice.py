import tensorflow as tf


def infer(self, image=None):
    tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor_image = self.preprocess(tensor_image)
    shape = tensor_image.shape
    tensor_image = tf.reshape(tensor_image, [1, shape[0], shape[1], shape[2]])
    return self.predict(tensor_image)['conv2d_transpose_4']
