import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

class CNN():
  def __init__(self):
    model_directory = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    if not os.path.exists(model_directory):
      vggmodel = VGG16(weights= 'imagenet')
    else:
      vggmodel = VGG16(weights= model_directory)
    self.model = tf.keras.models.Model(vggmodel.input, vggmodel.layers[-2].output)

  def __preprocess_frames(self, frames):
    frames = np.array(list(map(lambda x: cv2.resize(x, (224,224)), frames)))
    preprocessed_frames = preprocess_input(frames)
    return preprocessed_frames
  
  def extract_features(self, frames):
    preprocessed_frames = self.__preprocess_frames(frames)
    features = self.model.predict(preprocessed_frames)
    return features