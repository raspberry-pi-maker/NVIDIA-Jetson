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
import sys
import time
import numpy as np
import cv2
# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("--network", type=str, default="facenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

'''
extra_rate = 1.0 ~ 1.X to enlarge the mask size
'''
def do_masking(detection, extra_rate = 1.0):
    global orgimg
    bear = mask.copy()
    oh, ow, _ = orgimg.shape
    w = min(int(detection.Width * extra_rate) , ow)
    h = min(int(detection.Height * extra_rate), oh)
    x = max(int(detection.Left - (w * (extra_rate - 1.0 ) / 2)) , 0)
    y = max(int(detection.Top  - (h * (extra_rate - 1.0 ) / 2)) , 0)

    dim = (w, h) 
    bear = cv2.resize(bear, dim, interpolation = cv2.INTER_AREA)
    bg = orgimg[y:y+h, x:x+w]
    print(bear.shape)
    for i in range(0, h):
        for j in range(0, w):
            B = bear[i][j][0]
            G = bear[i][j][1]
            R = bear[i][j][2]
            if (int(B) + int(G) + int(R)):
                bg[i][j][0] = B
                bg[i][j][1] = G
                bg[i][j][2] = R
    orgimg[y:y+h, x:x+w] = bg


# load an image (into shared CPU/GPU memory)
print('input image:%s  output image:%s'%(opt.file_in, opt.file_out))
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)
orgimg = jetson.utils.cudaToNumpy(img, width, height, 4)      #CUDA img is float type
orgimg = cv2.cvtColor (orgimg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
mask = cv2.imread('./sbear.png', cv2.IMREAD_UNCHANGED)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
t = time.time()
# detect objects in the image (with overlay)
detections = net.Detect(img, width, height, opt.overlay)
elapsed = time.time() - t
# print the detections
print("detected {:d} objects in image".format(len(detections)))
print("FPS:%f"%(1.0 / elapsed))

for detection in detections:
	print(detection)


# print out timing info
net.PrintProfilerTimes()

jetson.utils.cudaDeviceSynchronize ()
arr = jetson.utils.cudaToNumpy(img, width, height, 4)      #CUDA img is float type
arr = cv2.cvtColor (arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
for detection in detections:
    do_masking(detection, extra_rate = 1.4)

if opt.file_out is not None:
    cv2.imwrite("/tmp/facedetect.jpg", arr)
    cv2.imwrite("/tmp/facemask.jpg", orgimg)

