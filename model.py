import cv2
import numpy as np
import pickle
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from transformers import pipeline


class model_platDetection():
  def __init__(self, model_path:str= "model\plate_detection_skiba4") -> None:
    self.model_path = model_path
    self.__pipe = pipeline("object-detection", model=self.model_path)
    
  def predict(self, img_path:str):
    return self.__pipe(img_path)
  

class model_character():
  max_len = 8
  print("processing....")
  def __init__(self, model_path:str = "model\model_LPR",model_labelEncoder_path:str = "model\model_LabelEncoder\labelEncoder.sav") -> None:
    self.model_path = model_path
    self.model_labelEncoder = pickle.load(open(model_labelEncoder_path, "rb"))
    self.model = self.__build_model()
  # y = np.array([le.transform(i) for i in y])


  def __preprocess_img(self, image_path):
      img = tf.io.read_file(image_path)
      img = tf.io.decode_jpeg(img, channels=1)
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = tf.image.resize(img, [224, 224])
      img = tf.transpose(img, perm=[1, 0, 2])
      
      return img


  def __build_model(self):
    loaded_model = tf.keras.models.load_model(self.model_path)
    
    prediction_model_loaded = tf.keras.models.Model(
        loaded_model.get_layer(name="image").input, 
        loaded_model.get_layer(name="dense2").output
    )
    
    return prediction_model_loaded

  def predict_image(self, image_path):
    image = self.__preprocess_img(image_path)
    pred = self.model.predict(tf.expand_dims(image, axis=0))
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]
    results = np.squeeze(results)
    convert_result = np.array([self.model_labelEncoder.inverse_transform([i]) for i in results if i != -1])

    return np.squeeze(convert_result)