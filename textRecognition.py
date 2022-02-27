import os
import cv2
import numpy as np
from textRecognitionModel import Model
import keras

char_list = ['a', 'A', 'b', 'B', 'c', 'C', 'ç', 'Ç', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'ğ', 'Ğ', 'h', 'H', 'ı', 'I', 'i', 'İ', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ö', 'Ö', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 'ş', 'Ş', 't', 'T', 'u', 'U', 'ü', 'Ü', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', ',', '.', '#', '+', '%', ';', '=', '(', ')', "'"]


def recognition(model_weight_path):
    prediction_model = Model()
    prediction_model.load_weights(model_weight_path)
    for k in os.listdir("/content/ConnectModels/yolo_crop"):
      fileName = "/content/ConnectModels/yolo_crop/" + k
      img = cv2.imread(fileName,0)
      img = cv2.resize(img,(128,32))

      preds = prediction_model.predict(img.reshape(-1,32,128,1),batch_size=1)
      input_len = np.ones(preds.shape[0]) * preds.shape[1]
      results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]

      res = "".join([char_list[i] for i in results[0] if i != -1 and i < len(char_list)])

      print(res)
    
