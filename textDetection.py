import cv2
import numpy as np
import tensorflow as tf

def detect_text(img,model,target_path):

  original_image = img.copy()
  cropImage = img.copy()

  # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.resize(img,(608,608))
  img = np.reshape(img,(-1,608,608,3))

  images_data = []
  images_data = np.reshape(img,(1,608,608,3))
  images_data = images_data / 255.
  images_data = np.asarray(images_data).astype(np.float32)

  batch_data = tf.constant(images_data)

  pred_bbox = model.predict(batch_data)


  boxes = pred_bbox[:, :, 0:4]
  pred_conf = pred_bbox[:, :, 4:]

  boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
      boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
      scores=tf.reshape(
          pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
      max_output_size_per_class=1000,
      max_total_size=1000,
      iou_threshold=0.2,
      score_threshold=0.3
  )

  pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

  out_boxes, out_scores, out_classes, num_boxes = pred_bbox

  wordList = []

  imgH, imgW, _ = original_image.shape
  for i in range(num_boxes[0]):
    

    coor = out_boxes[0][i]
    x1 = int(coor[0] * imgH)
    x2 = int(coor[2] * imgH)
    y1 = int(coor[1] * imgW)
    y2 = int(coor[3] * imgW)

    c1, c2 = (y1, x1), (y2, x2)
    
    cv2.rectangle(original_image, c1, c2, (0,255,0), 2)
    cropped = cropImage[x1:x2,y1:y2]

    name = str(int(y1))+" "+str(int(x1))+" "+str(int(y2))+" "+str(int(x2))

    wordList.append([cropped,name])

    # cv2.imwrite(target_path+"/"+str(name)+".jpg",cropped)

  return original_image, wordList



