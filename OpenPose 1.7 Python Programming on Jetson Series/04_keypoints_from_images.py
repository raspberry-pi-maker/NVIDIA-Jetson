# Customized for NVidia Jetson Series by spyjetson
# From Python
# It requires OpenCV installed for Python

import sys
import cv2
import os
from sys import platform
import argparse
import time
from openpose import pyopenpose as op

try:

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/usr/local/src/openpose-1.7.0/examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--save_dir", default="/usr/local/src/output/", help="directory to save the output images")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["net_resolution"] = "320x320" 

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

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    # Process and display images
    for imagePath in imagePaths:
        filename = os.path.basename(imagePath)    
        fname, extension = os.path.splitext(filename)
        if(extension is None or (extension != '.jpg' and extension != '.png')):
            print('File[%s] is not a image file'%imagePath)
            continue
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        print("Body keypoints: \n" + str(datum.poseKeypoints))

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(100)
            if key == 27: break
        save_file = os.path.join(args[0].save_dir, fname + '_rendered' + extension)
        cv2.imwrite(save_file, datum.cvOutputData)

        
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
