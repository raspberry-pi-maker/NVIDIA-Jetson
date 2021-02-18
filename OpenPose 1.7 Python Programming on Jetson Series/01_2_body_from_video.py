# Customized for NVidia Jetson Series by spyjetson
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os, time
from sys import platform
import argparse
from openpose import pyopenpose as op
from datetime import datetime

try:
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/usr/local/src/openpose-1.7.0/examples/media/video.avi", help="Process an video. ")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["net_resolution"] = "320x-1" 
    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    # Process video
    cap = cv2.VideoCapture(args[0].video_path)
    color = (0,0,255) #BGR
    thickness = -1      #draw inner space of circle
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        s = datetime.now() 
        ret,img = cap.read()
        if ret == False:
            break
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        human_count = len(datum.poseKeypoints)
        # Display Image
        
        for human in range(human_count):
            for j in range(25):
                if datum.poseKeypoints[human][j][2] > 0.01:
                    center = (int(datum.poseKeypoints[human][j][0]) ,  int(datum.poseKeypoints[human][j][1]))
                    cv2.circle(img, center, 3, color, thickness)
        e = datetime.now()
        delta = e - s
        sec = delta.total_seconds()   
        
        cv2.putText(img,'FPS[%5.2f] %d person detected'%(1/( sec),human_count),(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
        cv2.waitKey(1)

except Exception as e:
    print(e)
    sys.exit(-1)
    
cap.release()
cv2.destroyAllWindows()
    
