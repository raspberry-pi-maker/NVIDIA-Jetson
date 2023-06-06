'''
https://github.com/yas-sim/object-tracking-line-crossing-area-intrusion/tree/master 참조
pip install opencv-python numpy scipy munkres 
'''
import cv2
from ultralytics import YOLO
import numpy as np
from line_boundary_check import *
import argparse

class boundaryLine:
    def __init__(self, line=(0,0,0,0)):
        self.p0 = (line[0], line[1])
        self.p1 = (line[2], line[3])
        self.color = (0,255,255)
        self.lineThinkness = 2
        self.textColor = (0,255,255)
        self.textSize = 2
        self.textThinkness = 2
        self.count1 = [0,0]   #person, vehicles
        self.count2 = [0,0]



# Draw single boundary line
def drawBoundaryLine(img, line):
    x1, y1 = line.p0
    x2, y2 = line.p1
    cv2.line(img, (x1, y1), (x2, y2), line.color, line.lineThinkness)
    cv2.putText(img, "person:" + str(line.count1[0]), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.putText(img, "vehicles:" + str(line.count1[1]), (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)

    cv2.putText(img, "person:" + str(line.count2[0]), (x2 - 100, y2 + 30), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.putText(img, "vehicles:" + str(line.count2[1]), (x2 - 100, y2 + 50), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.drawMarker(img, (x1, y1),line.color, cv2.MARKER_TRIANGLE_UP, 16, 4)
    cv2.drawMarker(img, (x2, y2),line.color, cv2.MARKER_TILTED_CROSS, 16, 4)


# Draw multiple boundary lines
def drawBoundaryLines(img, boundaryLines):
    for line in boundaryLines:
        drawBoundaryLine(img, line)

# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
def checkLineCross(boundary_line, cls, trajectory_line):
    traj_p0  = trajectory_line[0]                                       # Trajectory of an object
    traj_p1  = trajectory_line[1]
    bLine_p0 = (boundary_line.p0[0], boundary_line.p0[1])               # Boundary line
    bLine_p1 = (boundary_line.p1[0], boundary_line.p1[1])
    intersect = checkIntersect(traj_p0, traj_p1, bLine_p0, bLine_p1)    # Check if intersect or not
    if intersect == True:
        angle = calcVectorAngle(traj_p0, traj_p1, bLine_p0, bLine_p1)   # Calculate angle between trajectory and boundary line
        if angle<180:
            if(cls == 1):
                boundary_line.count1[0] += 1
            else:    
                boundary_line.count1[1] += 1
        else:
            if(cls == 1):
                boundary_line.count2[0] += 1
            else:    
                boundary_line.count2[1] += 1


def update_vehicles(data):
    global active_vehicles
    for k, v in active_vehicles.items():    
        active_vehicles[k][0] = False
    for box in data:
        cx = int((box[0].item() + box[2].item()) / 2)
        cy = int((box[1].item() + box[3].item()) / 2)
        id = int(box[4].item())
        if id in active_vehicles:
            active_vehicles[id][0] = True
            active_vehicles[id][1] = active_vehicles[id][2]
            active_vehicles[id][2] = [cx, cy]
        else:
            active_vehicles[id] = [True, [cx, cy], [cx, cy]]

    #remove invalid vehicles
    del_list = []    
    for k, v in active_vehicles.items():    
        if active_vehicles[k][0] == False:
            del_list.append(k)

    for k in del_list:
        del active_vehicles[k]




# boundary lines
boundaryLines = [
    boundaryLine([ 0, 220,  950, 300 ])
]
label_map = None

active_vehicles = {}

def main():
    global label_map, active_vehicles

    parser = argparse.ArgumentParser()
    parser.add_argument('--track',  type=str, default="bytetrack.yaml" )  #botsort.yaml or bytetrack.yaml
    args = parser.parse_args()

    colors = [(255,255 , 0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX   
    model = YOLO("yolov8m.pt")
    model.to('cuda')
    label_map = model.names

    for result in model.track(source="./highway_traffic.mp4", show=False, stream=True, agnostic_nms=False, conf= 0.1,  tracker=args.track):
        update_vehicles(result.boxes.data)
        frame = result.orig_img
        for box, conf, cls in zip(result.boxes.data, result.boxes.conf, result.boxes.cls):
            p1 =  (int(box[0].item()), int(box[1].item()))
            p2 =  (int(box[2].item()), int(box[3].item()))
            id = int(box[4].item())
            cls = int(cls.item())
            cv2.rectangle(frame, p1, p2, colors[int(cls % len(colors))], 2)
            text = "#" + str(id) + "-"+ label_map[cls] + " %4.2f"%(conf.item()) 
            cv2.putText(frame, text, (p1[0], p1[1] - 10), font, fontScale = 1, color = colors[int(cls % len(colors))], thickness = 2)


            for line in boundaryLines:
                checkLineCross(line, cls, (active_vehicles[id][1], active_vehicles[id][2]))            

        drawBoundaryLines(frame, boundaryLines)
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(1) == 27):
            break


if __name__ == "__main__":
    main()