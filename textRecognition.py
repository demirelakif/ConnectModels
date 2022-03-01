import os
import cv2
import numpy as np
from textRecognitionModel import Model
import keras
#
char_list = ['a', 'A', 'b', 'B', 'c', 'C', 'ç', 'Ç', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'ğ', 'Ğ', 'h', 'H', 'ı', 'I', 'i', 'İ', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ö', 'Ö', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 'ş', 'Ş', 't', 'T', 'u', 'U', 'ü', 'Ü', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', ',', '.', '#', '+', '%', ';', '=', '(', ')', "'"]


def recognition(prediction_model,crop_path):
    images = []
    words = []
    for k in os.listdir(crop_path):
      fileName = crop_path +"/" + k
      img = cv2.imread(fileName,0)
      img = cv2.resize(img,(128,32))
      images.append(img)
      coordinates = k.split(".")[0]
      
      coordinates = coordinates.split(" ")
      x , y , w ,h  = coordinates
      x , y , w ,h = int(x) , int(y), int(w), int(h)
      minY, maxY, minX, maxX = y , y+h , x , x + w
      words.append([minY,minX,maxY,maxX])
    
    images = np.array(images)

    preds = prediction_model.predict(images.reshape(-1,32,128,1),batch_size=64)
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
    
    for k in range(len(results)):

      res = "".join([char_list[i] for i in results[k] if i != -1 and i < len(char_list)])
      #hiç tahmin yapılamamışsa
      try:
        if len(res) > 0:
          words[k] = words[k] + [res,]
        else:
          words[k] = words[k] + [" ",]
      except:
        pass
    
    # print(words)
    # for i in words:
    #   print(i[4])
    #   if i[4] == None :
    #     pass
      
    lines = []
    temp = []
    # Aynı hizada olan kelimeleri bulup satırların tutulduğu listeye ekliyor.
    for i in range(len(sorted(words)) - 1):
      minY,minX,maxY,maxX, word0 = sorted(words)[i]

      y0 = int(maxY - ((maxY - minY)/2))

      minY,minX,maxY,maxX, word1 = sorted(words)[i + 1]
      avgwordHeight = (maxY - minY) * 3 / 5

      y1 = int(maxY - ((maxY - minY)/2))
          
      if (y1 - y0) < avgwordHeight:
        temp.append(sorted(words)[i])
      else:
        temp.append(sorted(words)[i])
        lines.append(temp.copy())
        temp = []

    newText = ""

    # Kelimeler satır satır newText stringine ekleniyor.
    for line in lines:
      for i in range(len(line)):
          line[i][0] = line[0][0]
      
      text = ""
      
      avgX = [int((i[3] + i[1])/2) for i in line]

      
      for i in range(len(line)):
          minY,minX,maxY,maxX, word = line[avgX.index(sorted(avgX)[i])]
          text += word + " "
          
      text = text.strip()
      newText += text + "\n"
    #print(newText)

    return newText
    

     










# import os
# import cv2
# import numpy as np
# from textRecognitionModel import Model
# import keras

# char_list = ['a', 'A', 'b', 'B', 'c', 'C', 'ç', 'Ç', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'ğ', 'Ğ', 'h', 'H', 'ı', 'I', 'i', 'İ', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ö', 'Ö', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 'ş', 'Ş', 't', 'T', 'u', 'U', 'ü', 'Ü', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', ',', '.', '#', '+', '%', ';', '=', '(', ')', "'"]


# def recognition(model_weight_path,crop_path):
#     prediction_model = Model()
#     prediction_model.load_weights(model_weight_path)
#     words = []
#     for k in os.listdir(crop_path):
#       fileName = crop_path +"/" + k
#       img = cv2.imread(fileName,0)
#       img = cv2.resize(img,(128,32))

#       preds = prediction_model.predict(img.reshape(-1,32,128,1),batch_size=1)
#       input_len = np.ones(preds.shape[0]) * preds.shape[1]
#       results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
#       coordinates = k.split(".")[0]
      
#       coordinates = coordinates.split(" ")
#       x , y , w ,h  = coordinates
#       x , y , w ,h = int(x) , int(y), int(w), int(h)
#       minY, maxY, minX, maxX = y , y+h , x , x + w
#       res = "".join([char_list[i] for i in results[0] if i != -1 and i < len(char_list)])
#       words.append([minY,minX,maxY,maxX,res])


#     lines = []
#     temp = []
#     # Aynı hizada olan kelimeleri bulup satırların tutulduğu listeye ekliyor.
#     for i in range(len(sorted(words)) - 1):
#       minY,minX,maxY,maxX, word0 = sorted(words)[i]

#       y0 = int(maxY - ((maxY - minY)/2))

#       minY,minX,maxY,maxX, word1 = sorted(words)[i + 1]
#       avgwordHeight = (maxY - minY) * 3 / 5

#       y1 = int(maxY - ((maxY - minY)/2))
          
#       if (y1 - y0) < avgwordHeight:
#         temp.append(sorted(words)[i])
#       else:
#         temp.append(sorted(words)[i])
#         lines.append(temp.copy())
#         temp = []

#     newText = ""

#     # Kelimeler satır satır newText stringine ekleniyor.
#     for line in lines:
#       for i in range(len(line)):
#           line[i][0] = line[0][0]
      
#       text = ""
      
#       avgX = [int((i[3] + i[1])/2) for i in line]

      
#       for i in range(len(line)):
#           minY,minX,maxY,maxX, word = line[avgX.index(sorted(avgX)[i])]
#           text += word + " "
          
#       text = text.strip()
#       newText += text + "\n"
#     print(newText)
