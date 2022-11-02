# from yolov7.detect import DetectObjects
import torch 
import cv2
from yolov7.detect import DetectObjects
from bytetrack.byte_tracker import BYTETracker




with torch.no_grad():
    yolo = DetectObjects('./models/yolov7.pt',640,0.25)
    tracker = BYTETracker()

    img = cv2.imread('./yolov7/inference/images/horses.jpg')
    dets = yolo.predict(img)
    print(dets)
    # print(tracker.update(dets, ))
