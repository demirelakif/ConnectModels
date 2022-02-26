import cv2
import numpy as np

def predict(image):

    net = cv2.dnn.readNetFromDarknet("yolov3_custom_best.weights","yolov3_custom.cfg")
    classes = ["text"]
    print(image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    print(img.shape)
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (512, 512), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():

        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # print(len(boxes))
        resm = cv2.imread(image)
        for k, j in enumerate(boxes):
            x, y, w, h = j
            # print(x,y,w,h)
            cropped = resm[y:y+h, x:x+w]
            try:
                cv2.imwrite("yolo_crop/"+str(k)+".jpg", cropped)
                # print(cropped)

            except:
                pass
    cv2.imwrite("textDetection.jpg", img)



