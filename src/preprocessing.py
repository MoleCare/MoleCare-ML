import tensorflow as tf
import keras.applications

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8),))
def preprocessing(input_tensor):
    output_tensor = tf.cast(input_tensor, dtype=tf.float32)
    output_tensor = tf.image.resize_with_pad(output_tensor, target_height=224, target_width=224)
    output_tensor = keras.applications.mobilenet.preprocess_input(output_tensor, data_format="channels_last")
    return output_tensor