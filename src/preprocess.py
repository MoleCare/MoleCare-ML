import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pathlib

class Preprocessing :
  staticVar = 'hi' #class variable shared by all instances

  def __init__(self):
    self.imgHeight = 224 #instance variable unique to each instance
    self.imgWidth = 224
    self.testImage = None
    self.mlModel = None
  
  def __init__(self, imgHeight, imgWidth):
    self.imgHeight = imgHeight
    self.imgWidth = imgWidth
    self.testImage = None
    self.mlModel = None

  def loadModel(self) :
    # Recreate the exact same model, including its weights and the optimizer
    self.mlModel = tf.keras.models.load_model('./../ml-model/model_molecare_v1.h5')

  def getModel(self) :
    return self.mlModel

  def printModel(self) :
    if self.mlModel != None :
      print(self.mlModel.summary())

  def loadTestImage(self) :    
    pathImg = pathlib.Path("../data/static/test_images/Melanoma/ISIC_0034074.jpg")
    testImage = tf.keras.preprocessing.image.load_img(pathImg, target_size=(self.imgHeight, self.imgWidth))

    arrTestImage = tf.keras.preprocessing.image.img_to_array(testImage)
    arrTestImage = tf.expand_dims(arrTestImage, 0)  # Create batch axis
    nArrTestImage = self.normalize(arrTestImage)

    predictions = self.mlModel.predict(nArrTestImage)

    score = 100 * predictions[0]
    print(
        "This image is %.2f percent Melanoma"
        % score
    )
    return score

  def normalize(image):
      image = tf.cast(image/255., tf.float32)
      return image
