import cv2

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ["text"]

#

def detect_text(image,model,save_path):

    #height , width , _ = image.shape
    crop_img = image
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)





    for k,(classid, score, box) in enumerate(zip(classes, scores, boxes)):
        (x,y,w,h) = box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        try:
          cropped = crop_img[y-3:y+h+3, x-3:x+w+3]
          name = str(int(x))+" "+str(int(y))+" "+str(int(w))+" "+str(int(h))
          cv2.imwrite(save_path+"/"+str(name)+".jpg", cropped)
        except:
          pass
        



#img = cv2.resize(img,(512,512))
#cv2.imshow("detections", img)
#cv2.waitKey(0)
