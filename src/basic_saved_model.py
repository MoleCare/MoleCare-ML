"""
import tensorflow as tf

from src.decode_and_serve import decode_and_serve
from src.preprocessing import preprocessing

tf.saved_model.save(
    classifier,
    export_dir="classifier/1",
    signatures={
        "serving_default": decode_and_serve,
        "preprocessing": preprocessing,
    },
)
"""