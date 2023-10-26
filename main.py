import cv2
import numpy as np

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


max_len = 8


base_character = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(base_character)
print(le.classes_)
# y = np.array([le.transform(i) for i in y])


def preprocess_img(image_path):
    img = tf.io.read_file(image_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # print(img)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [224, 224])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])

    return img

loaded_model = tf.keras.models.load_model("model\model_LPR")

prediction_model_loaded = tf.keras.models.Model(
    loaded_model.get_layer(name="image").input, loaded_model.get_layer(name="dense2").output
)

def predict_image(image, model):
  pred = model.predict(tf.expand_dims(image, axis=0))
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
  results = np.squeeze(results)
  convert_result = np.array([le.inverse_transform([i]) for i in results if i != -1])

  return np.squeeze(convert_result)


img_test = preprocess_img("dataset\platGray\E536YY.jpg")

print(predict_image(img_test, prediction_model_loaded))