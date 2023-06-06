import cv2
import argparse
from ultralytics import YOLO
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track',  type=str, default="botsort.yaml" )  #botsort.yaml or bytetrack.yaml
    args = parser.parse_args()
    model = YOLO("yolov8m.pt")
    model.to('cuda')

    colors = [(255,255 , 0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX   
    label_map = model.names

    for result in model.track(source="./sample.mp4", show=False, stream=True, agnostic_nms=True,  tracker=args.track):
        
        frame = result.orig_img

        for box, conf, cls in zip(result.boxes.data, result.boxes.conf, result.boxes.cls):
            index = 0
            p1 =  (int(box[0].item()), int(box[1].item()))
            p2 =  (int(box[2].item()), int(box[3].item()))
            id = int(box[4].item())
            cv2.rectangle(frame, p1, p2, colors[int(cls.item() % len(colors))], 2)
            text = "#" + str(id) + "-"+ label_map[int(cls.item())] + " %4.2f"%(conf.item()) 
            cv2.putText(frame, text, (p1[0], p1[1] - 10), font, fontScale = 1, color = colors[int(cls.item() % len(colors))], thickness = 2)
            index += 1

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(1) == 27):
            break


if __name__ == "__main__":
    main()