import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pathlib

#model = tf.saved_model.load('./tf_model')
#model = load_model('./tf_model')

# Recreate the exact same model, including its weights and the optimizer
model_v1 = tf.keras.models.load_model('model_molecare_v1.h5')
# Show the model architecture
model_v1.summary()
print(model_v1)

path_img = pathlib.Path("./test_images/Melanoma/ISIC_0034074.jpg")

img_height = 224
img_width = 224

test_img = tf.keras.preprocessing.image.load_img(
    path_img, target_size=(img_height, img_width)
)

img_test1 = tf.keras.preprocessing.image.img_to_array(test_img)
img_test1 = tf.expand_dims(img_test1, 0)  # Create batch axis
img_test1 = iimage = tf.cast(img_test1/255. ,tf.float32)

predictions = model_v1.predict(img_test1)
score = predictions[0]
print(
    "This image is %.2f percent Melanoma"
    % (100 * score)
)
