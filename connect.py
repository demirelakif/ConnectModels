import time
import cv2
from absl.flags import FLAGS
from receiptDetection import detectReceipts
from textRecognition import recognition
from detectionYolov4 import detect_text
import time
from absl import flags,app

flags.DEFINE_string('receipt_weights_path', '/content/500_loss_0.08_iou_0.97.h5',
                    'path to receipt weights file')

flags.DEFINE_string('yolo_weights_path', '/content/yolov4_custom_best.weights',
                    'path to weights file')

flags.DEFINE_string('yolo_cfg_path', '/content/yolov4_custom.cfg',
                    'path to cfg file')

flags.DEFINE_string('recognition_weights_path', '/content/',
                    'path to recognition weights file')

flags.DEFINE_string('image', '',
                    'image file')

def main(args):

    resm = FLAGS.image
    name = resm.split(".")[0]

    receipt_weights_path = FLAGS.receipt_weights_path
    yolo_weight_path = FLAGS.yolo_weights_path
    yolo_cfg_path = FLAGS.yolo_cfg_path
    recognition_weights_path = FLAGS.recognition_weights_path


    
    img = cv2.imread(resm)
    start_time = time.time()
    detectReceipts(img,name,receipt_weights_path)
    print("--- receipt detection %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    detect_text(resm,yolo_weight_path,yolo_cfg_path)
    print("--- text detection %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    recognition(recognition_weights_path)
    print("--- text recognition %s seconds ---" % (time.time() - start_time))



app.run(main)