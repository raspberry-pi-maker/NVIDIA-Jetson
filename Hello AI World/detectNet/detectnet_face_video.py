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

parser.add_argument("--video_in", type=str, help="filename of the input video to process")
parser.add_argument("--video_out", type=str, default=None, nargs='?', help="filename of the output video to save")
parser.add_argument("--network", type=str, help="pre-trained model to load (facenet)")
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
def do_masking(detection, arr, extra_rate = 1.0):
    bear = mask.copy()
    oh, ow, _ = arr.shape
    w = min(int(detection.Width * extra_rate) , ow)
    h = min(int(detection.Height * extra_rate), oh)
    x = max(int(detection.Left - (w * (extra_rate - 1.0 ) / 2)) , 0)
    y = max(int(detection.Top  - (h * (extra_rate - 1.0 ) / 2)) , 0)

    dim = (w, h) 
    bear = cv2.resize(bear, dim, interpolation = cv2.INTER_AREA)
    bg = arr[y:y+h, x:x+w]
    print(bear.shape)
    try:
        for i in range(0, h):
            for j in range(0, w):
                B = bear[i][j][0]
                G = bear[i][j][1]
                R = bear[i][j][2]
                if (int(B) + int(G) + int(R)):
                    bg[i][j][0] = B
                    bg[i][j][1] = G
                    bg[i][j][2] = R
        arr[y:y+h, x:x+w] = bg
    except IndexError:
        print(' index Error')
        return None
    
    return arr


def process_frame(img):
    global out_video
    height, width, _ = img.shape
    cudaimg = cv2.cvtColor (img.astype(np.uint8), cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson.utils.cudaFromNumpy(cudaimg)
    detections = net.Detect(cuda_mem, width, height, opt.overlay)
    jetson.utils.cudaDeviceSynchronize ()
    # arr = jetson.utils.cudaToNumpy(cuda_mem, width, height, 4)      #CUDA img is float type
    # arr = cv2.cvtColor (arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    for detection in detections:
        img = do_masking(detection, img, extra_rate = 1.2)
        if img is None:
            return False
    out_video.write(img)
    elapsed = time.time() - t
    # print the detections
    print("Frame[%d] FPS:%f"%(count, 1.0 / elapsed))
    return True


cap = cv2.VideoCapture(opt.video_in)
ret, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter(opt.video_out, fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))
mask = cv2.imread('./sbear.png', cv2.IMREAD_UNCHANGED)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

count = 0
while cap.isOpened():
    count += 1
    t = time.time()
    ret, img = cap.read()
    if ret == False:
        break
    if False == process_frame(img):
        break

out_video.release()
cap.release()