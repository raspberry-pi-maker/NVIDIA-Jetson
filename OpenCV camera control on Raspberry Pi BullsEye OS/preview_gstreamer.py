import cv2
import numpy as np
import sys
connstr = 'libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! clockoverlay time-format="%D %H:%M:%S" ! appsink'
cap = cv2.VideoCapture(connstr, cv2.CAP_GSTREAMER)
if cap.isOpened() == False:
    print('camera open Failed')
    sys.exit(0)


while True:

    succes, img = cap.read()
    if succes == False:
        print('camera read Failed')
        sys.exit(0)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    cv2.imshow('Img',img)

cap.release()
cv2.destroyAllWindows()