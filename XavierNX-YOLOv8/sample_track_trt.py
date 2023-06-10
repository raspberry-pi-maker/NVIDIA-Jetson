'''
from https://github.com/mikel-brostrom/yolo_tracking/tree/master/boxmot
copy boxmot (where trackers resides)
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time, sys
from models import TRTModule  # isort:skip
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from config import CLASSES, COLORS
import argparse
import torch

from bytetrack.byte_tracker import BYTETracker

parser = argparse.ArgumentParser()
parser.add_argument('--track',  type=str, default="bytetrack" )  #At this point, only supports bytetrack
args = parser.parse_args()

colors = [(255,0 , 0), (0,255,0), (0,0,255)]
font = cv2.FONT_HERSHEY_SIMPLEX   
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

device = 'cuda:0'
engine = "yolov8s.engine"
# Load a model
Engine = TRTModule(engine, device)
H, W = Engine.inp_info[0].shape[-2:]
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

tracker = BYTETracker()
#label_map = model.names

f = 0
net_total = 0.0
total = 0.0

def draw(img, boxes):
    index = 0
    for box in boxes.data:
        p1 =  (int(box[0].item()), int(box[1].item()))
        p2 =  (int(box[2].item()), int(box[3].item()))
        img = cv2.rectangle(img, p1, p2, colors[index % len(colors)], 3)
        text = label_map[int(box[5].item())] + " %4.2f"%(box[4].item()) 
        cv2.putText(img, text, (p1[0], p1[1] - 10), font, fontScale = 1, color = colors[index % len(colors)], thickness = 2)
        index += 1
    # cv2.imshow("draw", img)
    # cv2.waitKey(1)
    out_video.write(img)



def main():
    global f, net_total, total
    cap = cv2.VideoCapture("./highway_traffic.mp4")
    # Skip first frame result
    ret, img = cap.read()
    h, w, c = img.shape
    img, ratio, dwdh = letterbox(img, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    data = Engine(tensor)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('./trt_track_result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))



    while cap.isOpened():
        s = time.time()
        ret, img = cap.read()
        if ret == False:
            break
            
        draw = img.copy()
        img, ratio, dwdh = letterbox(img, (W, H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)


        net_s = time.time()
        data = Engine(tensor)
        net_e = time.time()

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio
        track_data = []
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            track_data.append([bbox[0], bbox[1], bbox[2], bbox[3], score.item(), cls_id])
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)


        np_track = np.array(track_data)
        outputs = tracker.update(np_track, None)
        if outputs.size:
            for i in range(outputs.shape[0]):   #[][] 
                start = outputs[i][0:2]
                #print(start)
                end = outputs[i][2:2]
                track_id = outputs[i][4]
                coff = outputs[i][5]
                cls_id = outputs[i][6]
                cv2.putText(draw,
                            f'{track_id}', (int(start[0]) - 20, int(start[1]) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 0],
                            thickness=2)


        #cv2.imshow('result', draw)
        #cv2.waitKey(1)
        e = time.time()
        net_total += (net_e - net_s)
        total += (e - s)
        f += 1
        out_video.write(draw) # 
    
    fps = f / total 
    net_fps = f / net_total 

    print("Total processed frames:%d"%f)
    print("FPS:%4.2f"%fps)
    print("Net FPS:%4.2f"%net_fps)
    cv2.destroyAllWindows()
    cap.release()
    out_video.release()

if __name__ == "__main__":
    main()