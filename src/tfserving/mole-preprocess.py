import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

loaded_model = tf.keras.models.load_model('./ml-model/model_molecare_v1.h5')
# Show the model architecture
loaded_model.summary()
print(loaded_model)

model = tf.saved_model.load('./tf_model')

num_classes = 2 # Melanoma, NotMelanoma
p_same = 'same'
a_leaky_relu = tf.nn.leaky_relu
k_size = 3

image_channels = 3 #RGB
# no changes, but to be sure that all images have the same size
img_height = 224
img_width = 224

#batch size: 32
batch_size = 32

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  image_channels)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model_v1 = Sequential([
  data_augmentation,

  layers.Conv2D(filters = 32, kernel_size = k_size, padding=p_same, activation = a_leaky_relu),
  layers.MaxPooling2D(2, 2),

  layers.Conv2D(filters = 64, kernel_size = k_size, padding=p_same, activation = a_leaky_relu),
  layers.MaxPooling2D(2, 2),

  layers.Conv2D(filters = 128, kernel_size = k_size, padding=p_same, activation = a_leaky_relu),
  layers.MaxPooling2D(2, 2),

  layers.Flatten(),
  layers.Dense(128),

  layers.Dense(1, activation='sigmoid')
])

def load_trained_model(weights_path):
   model = model_v1
   model.load_weights(weights_path)

def normalize(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label