import os
from matplotlib import image
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
import numpy as np
import time
import cv2
import helpers
import matplotlib.pyplot as plt
import perspective
import textDetection

def detectReceipts(image,name):


    # Modeli ve ağırlıkları yükler. (Ağırlık Path'ı değiştirilmeli)
    BACKBONE = 'mobilenetv2'
    model = sm.Unet(BACKBONE,input_shape=(512, 512, 3), encoder_weights=None,classes=1,activation='sigmoid')
    model.load_weights('500_loss_0.08_iou_0.97.h5')


    h,w,c = image.shape
    dim = (w, h)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512))
    image = np.expand_dims(image,axis = 0)

    # Tahmin kısmı (Tahmin süresini hesaplamak için bazı kodlar eklenmiştir)
    start_time = time.time()
    pr_mask = model.predict(image).round()
        
    print("--- receipt detection %s seconds ---" % (time.time() - start_time))
    img = helpers.get_segment_crop(image.squeeze(),0,pr_mask.squeeze())  
    pr_mask = pr_mask[..., 0].squeeze() * 255
    pr_mask_bgr = pr_mask.astype('uint8')
    pr_mask_bgr = cv2.cvtColor(pr_mask,cv2.THRESH_BINARY)
    pr_mask_bgr = cv2.resize(pr_mask_bgr, dim, interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #cv2.imwrite(i,img)
    cv2.imwrite(str("temp"+"/"+name+".jpg"),pr_mask_bgr)
    #cv2.imwrite("test_sonuc/"+i,img)


resm = "621.jpg"
name = resm.split(".")[0]

img = cv2.imread(resm)
detectReceipts(img,name)
perspective.perspective()

for i in os.listdir("perspective"):
    textDetection.predict("perspective/"+i)
