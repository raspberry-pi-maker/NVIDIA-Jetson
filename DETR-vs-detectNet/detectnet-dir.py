#!/usr/bin/python3
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
import sys, time, os

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the object detection network
t1 = time.time()
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
t2 = time.time()
print("======== Network Load time:%f"%(t2 - t1))


src_path = '/usr/local/src/test_images'
dest_path = '/usr/local/src/result'



for root, dirs, files in os.walk(src_path):
    for fname in files:
        full_fname = os.path.join(root, fname)
        img, width, height = jetson.utils.loadImageRGBA(full_fname)
        t1 = time.time()
        detections = net.Detect(img, width, height, opt.overlay)
        elapsed = time.time() - t1
        # print the detections
        print("detected {:d} objects in image".format(len(detections)))
        fps = 1.0 / elapsed
        print("FPS:%f"%(fps))
        for detection in detections:
            print(detection)

        # print out timing info
        net.PrintProfilerTimes()

        # save the output image with the bounding box overlays
        s = os.path.splitext(fname)[0]
        out_name = os.path.join(dest_path, s + '_detectnet_%s_%4.2f.jpg'%(opt.network, fps))
        jetson.utils.saveImageRGBA(out_name, img, width, height)


