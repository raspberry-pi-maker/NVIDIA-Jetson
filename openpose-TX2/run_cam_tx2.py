import logging
import sys
import time
import math
import cv2
import numpy as np
from openpose import pyopenpose as op

if __name__ == '__main__':
    fps_time = 0

    params = dict()
    params["model_folder"] = "../../models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    print("OpenPose start")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
      
    ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
    
    count = 0

    if cap is None:
        print("Camera Open Error")
        sys.exit(0)
    while cap.isOpened() and count < 30:
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break
        #dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", dst)
        #continue

        datum = op.Datum()
        datum.cvInputData = dst
        opWrapper.emplaceAndPop([datum])
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        newImage = datum.cvOutputData[:, :, :]
        cv2.putText(newImage , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_video.write(newImage)

        print("captured fps %f"%(fps))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", newImage)
        count += 1


    cv2.destroyAllWindows()        
    out_video.release()
    cap.release()








