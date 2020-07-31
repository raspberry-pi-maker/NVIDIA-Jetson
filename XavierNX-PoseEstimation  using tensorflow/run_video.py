import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')    
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640,480))
    count = 0
    t_netfps_time = 0
    t_fps_time = 0    

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    try:    
        while cap.isOpened():
            fps_time = time.time()
            ret_val, image = cap.read()
            if ret_val == False:
                print("Frame read End")
                break

            net_fps = time.time()
            humans = e.inference(image)
            netfps = 1.0 / (time.time() - net_fps)
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            t_netfps_time += netfps
            t_fps_time += fps
            cv2.putText(image, "NET FPS:%4.1f FPS:%4.1f" % (netfps, fps), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            img = cv2.resize(image, (640,480))            
            out_video.write(img) 
            #cv2.imshow('tf-pose-estimation result', image)
            print("captured fps[%f] net_fps[%f]"%(fps, netfps))

            if cv2.waitKey(1) == ord('q'):
                break
            count += 1    
    except KeyboardInterrupt:
        print("Keyboard interrupt exception caught")
        
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()
    if count:
        print("avg fps[%f] avg net_fps[%f]"%(t_fps_time / count, t_netfps_time / count)) 
logger.debug('finished+')
