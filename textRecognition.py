import os
import cv2
import numpy as np
from textRecognitionModel import Model
import keras

char_list = ['#', '%', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ı']


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



