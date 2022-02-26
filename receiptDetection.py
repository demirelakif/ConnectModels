import os
import numpy as np
import cv2
import helpers
from matplotlib import image
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
from perspective import perspective

def detectReceipts(image,name,weights_path):


    # Modeli ve ağırlıkları yükler. (Ağırlık Path'ı değiştirilmeli)
    BACKBONE = 'mobilenetv2'
    model = sm.Unet(BACKBONE,input_shape=(512, 512, 3), encoder_weights=None,classes=1,activation='sigmoid')
    model.load_weights(weights_path)


    h,w,c = image.shape
    dim = (w, h)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512))
    image = np.expand_dims(image,axis = 0)

    pr_mask = model.predict(image).round()
        
    img = helpers.get_segment_crop(image.squeeze(),0,pr_mask.squeeze())  
    pr_mask = pr_mask[..., 0].squeeze() * 255
    pr_mask_bgr = pr_mask.astype('uint8')
    pr_mask_bgr = cv2.cvtColor(pr_mask,cv2.THRESH_BINARY)
    pr_mask_bgr = cv2.resize(pr_mask_bgr, dim, interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #cv2.imwrite(i,img)
    cv2.imwrite(str("temp"+"/"+name+".jpg"),pr_mask_bgr)
    #cv2.imwrite("test_sonuc/"+i,img)
    perspective()