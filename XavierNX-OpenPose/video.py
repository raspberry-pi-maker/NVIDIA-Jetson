import sys
import time
import math
import cv2
import numpy as np
from openpose import pyopenpose as op
#import pyopenpose as op
import argparse

parser = argparse.ArgumentParser(description="OpenPose Example")
parser.add_argument("--video", type=str, required = True, help="video file name")
parser.add_argument("--res", type=str, default = "640x480", help="video file resolution")
args = parser.parse_args()
res = args.res.split('x')
res[0], res[1] = int(res[0]), int(res[1])
 

if __name__ == '__main__':
    fps_time = 0

    params = dict()
    params["model_folder"] = "/home/spypiggy/src/openpose/models/"
    params["net_resolution"] = args.res

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    print("OpenPose start")
    cap = cv2.VideoCapture(args.video)

    ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
    out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (res[0], res[1]))

    count = 0
    t_netfps_time = 0
    t_fps_time = 0
    if cap is None:
        print("Video[%s] Open Error"%(args.video))
        sys.exit(0)
    while cap.isOpened():
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Frame read End")
            break
        dst = cv2.resize(dst, dsize=(res[0], res[1]), interpolation=cv2.INTER_AREA)    

        datum = op.Datum()
        datum.cvInputData = dst
        net_fps = time.time()
        opWrapper.emplaceAndPop([datum])
        fps = 1.0 / (time.time() - fps_time)
        netfps = 1.0 / (time.time() - net_fps)
        t_netfps_time += netfps
        t_fps_time += fps
        
        fps_time = time.time()
        newImage = datum.cvOutputData[:, :, :]
        cv2.putText(newImage , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_video.write(newImage)

        print("captured fps[%f] net_fps[%f]"%(fps, netfps))
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", newImage)
        count += 1
    
    print("==== Summary ====")
    print("Inference Size : %s"%(args.res))
    if count:
        print("avg fps[%f] avg net_fps[%f]"%(t_fps_time / count, t_netfps_time / count))

    cv2.destroyAllWindows()
    out_video.release()
    cap.release()