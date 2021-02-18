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
    parser.add_argument("--image_path", default="/usr/local/src/openpose-1.7.0/examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose-1.7.0/models/"
    params["heatmaps_add_parts"] = True
    params["heatmaps_add_bkg"] = True
    params["heatmaps_add_PAFs"] = True
    params["heatmaps_scale"] = 2

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

    # Process outputs
    outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
    outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
    outputImageF = (outputImageF*255.).astype(dtype='uint8')
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    # Display Image
    counter = 0
    while 1:
        num_maps = heatmaps.shape[0]
        heatmap = heatmaps[counter, :, :].copy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", combined)
        key = cv2.waitKey(-1)
        if key == 27:
            break
        counter += 1
        counter = counter % num_maps
except Exception as e:
    print(e)
    sys.exit(-1)
