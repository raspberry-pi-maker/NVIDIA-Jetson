import cv2
import numpy as np
import sys
import time

connstr = None
f_res = None
def set_resolution(ver, index):
    global f_res, cap, connstr
    form = '''libcamerasrc ! video/x-raw, width={}, height={}, framerate=30/1 ! videoconvert ! videoscale ! appsink'''
    res = {"v1":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)],
           "v2":[(2592,1944),(1920, 1280),(1296, 972), (1296, 730),(640, 480)]
           }
    f_res = res[ver][index]
    connstr = form.format(f_res[0], f_res[1])
    print("GStreamer PipeLine:", connstr)
    

set_resolution("v1", 4)

cap = cv2.VideoCapture(connstr, cv2.CAP_GSTREAMER)
if cap.isOpened() == False:
    print('camera open Failed')
    sys.exit(0)

start = time.time()
while cap.isOpened():
    _, img = cap.read()
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    cv2.imshow('Img',img)
    elapsed = time.time() - start
    if(elapsed > 5) :
        break
cap.release()
cv2.destroyAllWindows()