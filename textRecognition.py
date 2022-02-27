import os
import cv2
import numpy as np
from textRecognitionModel import Model
import keras

char_list = ['a', 'A', 'b', 'B', 'c', 'C', 'ç', 'Ç', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'ğ', 'Ğ', 'h', 'H', 'ı', 'I', 'i', 'İ', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ö', 'Ö', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 'ş', 'Ş', 't', 'T', 'u', 'U', 'ü', 'Ü', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', ',', '.', '#', '+', '%', ';', '=', '(', ')', "'"]


def recognition(model_weight_path):
    images = []
    for i in os.listdir("yolo_crop"):
        img = cv2.imread("yolo_crop"+"/"+i,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(128,32))
        images.append(img)
    

    model = Model()
    model.load_weights(model_weight_path)


    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense").output
        )



    preds = prediction_model.predict(images.reshape(-1,32,128,1),batch_size=32)
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
    pred_texts = []

    for res in results.numpy():
        res = "".join([char_list[i] for i in res if i != -1 and i < len(char_list)])
        res = res.lower()
        pred_texts.append(res)


    for i in range(len(images)):
        pred = pred_texts[i].replace("[UNK]","")
        with open("preds.txt","a+") as f:
            f.write(pred+"\n")



