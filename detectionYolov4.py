import cv2

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ["text"]



def detect_text(image,yolo_weight_path,yolo_cfg_path):
    net = cv2.dnn.readNet(yolo_weight_path, yolo_cfg_path)
    #Cudnn aktifleştirilirse süre çok azalır
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    crop_img = cv2.imread(image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512, 512), scale=1/255, swapRB=True)
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)





    for k,(classid, score, box) in enumerate(zip(classes, scores, boxes)):
        color = COLORS[int(classid) % len(COLORS)]
        (x,y,w,h) = box
        cv2.rectangle(img, box, color, 2)
        print(box)
        cropped = crop_img[y-5:y+h+5, x-5:x+w+5]
        cv2.imwrite("yolo_crop/"+str(k)+".jpg", cropped)



#img = cv2.resize(img,(512,512))
#cv2.imshow("detections", img)
#cv2.waitKey(0)
