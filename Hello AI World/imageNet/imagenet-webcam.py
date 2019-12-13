#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys, time
import numpy as np
import cv2

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="/dev/video0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=640, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=480, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create the camera and display
font = jetson.utils.cudaFont()
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
#display = jetson.utils.glDisplay()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('/tmp/detect.mp4', fourcc, 25, (640, 480))
# process frames until user exits
count = 0
img, width, height = camera.CaptureRGBA(zeroCopy=1)
print("========== Capture Width:%d Height:%d ==========="%(width, height))

t = time.time()
while count < 500:
    # capture the image
    img, width, height = camera.CaptureRGBA(zeroCopy=1)
    # classify the image
    class_idx, confidence = net.Classify(img, width, height)
    # find the object description
    class_desc = net.GetClassDesc(class_idx)
    # overlay the result on the image	
    fps = 1.0 / ( time.time() - t)
    font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
    font.OverlayText(img, width, height, "FPS:%5.2f"%(fps), 5, 30, font.White, font.Gray40)
    t = time.time()
    #for numpy conversion, wait for synchronizing
    jetson.utils.cudaDeviceSynchronize ()
    arr = jetson.utils.cudaToNumpy(img, width, height, 4)      #CUDA img is float type
    arr1 = cv2.cvtColor (arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    if(count % 100 == 0):
        cv2.imwrite("/tmp/detect-" + str(count)+ ".jpg", arr1)
    out_video.write(arr1)
    cv2.imshow('imageNet', arr1)
    # print out performance info
    net.PrintProfilerTimes()
    count += 1

out_video.release()
