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
import cv2
import numpy as np
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
ret_val, img = cap.read()
height, width, _ = img.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0
fps_time = time.time()
if cap is None:
    print("Camera Open Error")
    sys.exit(0)

# process frames until user exits
while cap.isOpened() and count < 500:
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break    

    rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    img = cv2.UMat(img)
    # rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    #img = img.astype(np.float)
    # classify the image
    class_idx, confidence = net.Classify(img, width, height)
    # find the object description
    class_desc = net.GetClassDesc(class_idx)


    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()

    img = cv2.UMat.get(img) # GPU ->CPU
    # img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_video.write(newImage)


cv2.destroyAllWindows()        
out_video.release()
cap.release()