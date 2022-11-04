# from yolov7.detect import DetectObjects
import torch 
import cv2
import numpy as np
from yolov7.detect import DetectObjects
from byte.byte_tracker import BYTETracker
from yolov7.utils.plots import plot_one_box



with torch.no_grad():
    # Read Video capture
    cap    = cv2.VideoCapture('1_02.mp4')
    width  = cap. get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap. get(cv2.CAP_PROP_FRAME_HEIGHT)

    x_scale = width / 640
    y_scale = height / 640

    # Output Video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
    out_vid = cv2.VideoWriter("result_viedeo.mp4", fourcc, 30, (x_scale, y_scale),True)

    # Make instance of yolov7 and trackers
    yolo = DetectObjects('./yolov7.pt',640,0.25, device_name='cuda')
    tracker = BYTETracker(track_thresh = 0.15, track_buffer=30, frame_rate=30, match_thresh=0.6)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, org_frame = cap.read() 
        frame = cv2.resize(org_frame, (640,640))  

        # Detections 
        dets = yolo.predict(frame)

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
        
        # Save Traking output as image frame
        for idx, row in enumerate(online_tlwhs):
            plot_one_box(
                         (row[0]*x_scale, row[1]*y_scale, (row[0]+row[2])*x_scale, (row[1]+row[3])*x_scale),
                         org_frame, label=str(online_ids[idx]), 
                         color=yolo.colors[2], 
                         line_thickness=1
                         )
        
        out_vid.write(frame)
        # exit()







