import cv2
import sys

resolutions = [(1920, 1080), (1280,720), (1184,656),(1188,658),(1024,768), (640, 480), (800,600), (320,240), (320,176), (176,144), (160,120)]


print('Webcam resolution test start\n')

for resolution in resolutions:
    cap = cv2.VideoCapture(0, cv2.CAP_VFW)
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
        break
    ret1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    ret2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if(ret1 == True and ret2 == True):
        print('resolution [%d, %d] support success'%(resolution[0], resolution[1]))
        ret, img = cap.read()
        h, w, c = img.shape
        print('Video Frame shape W:%d, H:%d, Channel:%d\n'%(w, h, c))
    else:
        print('resolution [%d, %d] does not support '%(resolution[0], resolution[1]))
    cap.release()
