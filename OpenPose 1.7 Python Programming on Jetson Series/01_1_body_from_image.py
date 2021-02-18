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
    parser.add_argument("--image_path", default="/usr/local/src/openpose-1.7.0/examples/media/COCO_val2014_000000000294.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["net_resolution"] = "320x256" #COCO_val2014_000000000192.jpg image is landscape mode, so 320x256 is a good choice       
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

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    newImage = datum.cvOutputData[:, :, :]
    human_count = len(datum.poseKeypoints)
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    for human in range(human_count):
        for j in range(25):
            if datum.poseKeypoints[human][j][2] > 0.01:
                cv2.putText(newImage, str(j),  ( int(datum.poseKeypoints[human][j][0]) + 10,  int(datum.poseKeypoints[human][j][1])), font, 0.5, (0,255,0), 2) 
        print(datum.poseKeypoints[human])    
    
    # Display Image
    for human in range(human_count):
        print(datum.poseKeypoints[human])
    print("Total %d human detected"%human_count)
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", newImage)
    k = 0
    while k != 27:
        k = cv2.waitKey(0) & 0xFF
except Exception as e:
    print(e)
    sys.exit(-1)
