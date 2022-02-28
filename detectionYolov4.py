import cv2

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ["text"]



def detect_text(image,yolo_weight_path,yolo_cfg_path):
    net = cv2.dnn.readNet(yolo_weight_path, yolo_cfg_path)
    #Cudnn aktifleştirilirse süre çok azalır
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    #height , width , _ = image.shape
    crop_img = image
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512, 512), scale=1/255)
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)





    for k,(classid, score, box) in enumerate(zip(classes, scores, boxes)):
        (x,y,w,h) = box
        cropped = crop_img[y-3:y+h+3, x-3:x+w+3]
        name = str(int(x))+" "+str(int(y))+" "+str(int(w))+" "+str(int(h))
        cv2.imwrite("/content/ConnectModels/yolo_crop/"+str(name)+".jpg", cropped)



#img = cv2.resize(img,(512,512))
#cv2.imshow("detections", img)
#cv2.waitKey(0)
