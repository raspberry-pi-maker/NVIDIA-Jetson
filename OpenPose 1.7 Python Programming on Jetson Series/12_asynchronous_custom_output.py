# Customized for NVidia Jetson Series by spyjetson
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os, time
from sys import platform
import argparse
from openpose import pyopenpose as op


def display(datums):
    datum = datums[0]
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
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
        s = time.time()
        datumProcessed = op.VectorDatum()
        if opWrapper.waitAndPop(datumProcessed):
            if not args[0].no_display:
                # Display image
                userWantsToExit = display(datumProcessed)
            printKeypoints(datumProcessed)
        else:
            break

        e = time.time()
        print('FPS:%5.2f'%(1 / (e - s)))    
except Exception as e:
    print(e)
    sys.exit(-1)
