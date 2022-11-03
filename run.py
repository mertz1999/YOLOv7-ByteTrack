# from yolov7.detect import DetectObjects
import torch 
import cv2
import numpy as np
from yolov7.detect import DetectObjects
from byte.byte_tracker import BYTETracker
from yolov7.utils.plots import plot_one_box



with torch.no_grad():
    # Read Video capture
    cap = cv2.VideoCapture('1_01.mp4')

    # Make instance of yolov7 and trackers
    yolo = DetectObjects('./yolov7.pt',640,0.25)
    tracker = BYTETracker(track_thresh = 0.7, track_buffer=30, frame_rate=30, match_thresh=0.9)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read() 
        frame = cv2.resize(frame, (640,640))  

        # Detections 
        dets = yolo.predict(frame)

        # for row in dets[0]:
        #     plot_one_box((row[0],row[1],row[2],row[3]), frame, label=yolo.names[int(row[5])], color=yolo.colors[int(row[5])], line_thickness=1)
        
        # cv2.imshow("result",frame)

        #
        # online_targets = tracker.update(torch.tensor(dets[0]), (640,640), (640,640))
        # print(online_targets)
        # print(tracker.update(dets, ))

        # run tracking
        if dets[0] is not None:
            online_targets = tracker.update(torch.tensor(dets[0]), (640,640), (640,640))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 0.1 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            print(frame_id, online_tlwhs, online_ids, online_scores)

