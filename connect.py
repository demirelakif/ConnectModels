import os
import time
import cv2
import perspective
from textDetection import predict
from receiptDetection import detectReceipts

resm = "621.jpg"
name = resm.split(".")[0]

img = cv2.imread(resm)
detectReceipts(img,name)


