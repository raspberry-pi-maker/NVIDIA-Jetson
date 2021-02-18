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

def display(sec, datums):
    datum = datums[0]
    img = datum.cvInputData[:, :, :]
    human_count = len(datum.poseKeypoints)
    color = (0,0,255) #BGR
    thickness = -1      #draw inner space of circle
    font = cv2.FONT_HERSHEY_SIMPLEX

    for human in range(human_count):
        for j in range(25):
            if datum.poseKeypoints[human][j][2] > 0.01:
                center = (int(datum.poseKeypoints[human][j][0]) ,  int(datum.poseKeypoints[human][j][1]))
                cv2.circle(img, center, 3, color, thickness)

    cv2.putText(img,'FPS[%6.2f] %d person detected'%(1.0/( sec),human_count),(20,30), font, 1,(255,0,0),1,cv2.LINE_AA)
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
    key = cv2.waitKey(1)
    return (key == 27)


def printKeypoints(datums):
    datum = datums[0]
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    print("Face keypoints: \n" + str(datum.faceKeypoints))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))


try:
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", help="Disable display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["net_resolution"] = "320x256" 
    params["camera_resolution"] = "640x480" 

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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
    opWrapper.configure(params)
    opWrapper.start()

    # Main loop
    userWantsToExit = False
    while not userWantsToExit:
        # Pop frame
        s = datetime.now() 
        datumProcessed = op.VectorDatum()
        if opWrapper.waitAndPop(datumProcessed):
            e = datetime.now()
            delta = e - s
            sec = delta.total_seconds()    
            if not args[0].no_display:
                # Display image
                userWantsToExit = display(sec, datumProcessed)
            print('FPS:%6.2f Total [%d] frames return'%(1.0 / (sec), len(datumProcessed)))    
            #printKeypoints(datumProcessed)
        else:
            break

except Exception as e:
    print(e)
    sys.exit(-1)
