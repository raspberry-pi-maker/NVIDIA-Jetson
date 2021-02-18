# Customized for NVidia Jetson Series by spyjetson
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
from openpose import pyopenpose as op

try:

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="/usr/local/src/openpose-1.7.0/examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["net_resolution"] = "256x320" #COCO_val2014_000000000241.jpg image is portrait mode, so 256x320 is a good choice       
    params["face"] = True
    params["hand"] = True
    params["face_net_resolution"] = "256x256" # Multiples of 16 and squared default: "368x368"
    params["hand_net_resolution"] = "368x368" # Multiples of 16 and squared. default: "368x368" 

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
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    human_count = len(datum.poseKeypoints)

    # Display Image
    for human in range(human_count):
        print("%dth person body keypoints:\n"%human +  str(datum.poseKeypoints[human]))
        print("%dth person face keypoints:\n"%human +  str(datum.faceKeypoints[human]))
        print("%dth person Left Hand keypoints:\n"%human +  str(datum.handKeypoints[0][human]))
        print("%dth person Right Hand keypoints:\n"%human +  str(datum.handKeypoints[1][human]))

    print("Total %d human detected"%human_count)
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    k = 0
    while k != 27:
        k = cv2.waitKey(0) & 0xFF
except Exception as e:
    print(e)
    sys.exit(-1)
