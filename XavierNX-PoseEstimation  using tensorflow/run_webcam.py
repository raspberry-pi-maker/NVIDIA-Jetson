import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
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
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        w = 432
        h = 368
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cam.get(cv2.CAP_PROP_FPS), (640,480))

    count = 0
    t_netfps_time = 0
    t_fps_time = 0
    try:    
        while True:
            fps_time = time.time()
            ret_val, image = cam.read()

            logger.debug('image process+')
            net_fps = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            netfps = 1.0 / (time.time() - net_fps)
            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            #height, width, channels = img.shape
            print('Image shape:', image.shape)
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            logger.debug('show+')
            t_netfps_time += netfps
            t_fps_time += fps
            cv2.putText(image,
                        "NET FPS:%4.1f FPS:%4.1f" % (netfps, fps),
                        (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 5)
            img = cv2.resize(image, (640,480))            
            out_video.write(img)            
            print("captured fps[%f] net_fps[%f]"%(fps, netfps))
            #cv2.imshow('tf-pose-estimation result', image)
            if cv2.waitKey(1) == ord('q'):
                break
            logger.debug('finished+')
            count += 1
    except KeyboardInterrupt:
        print("Keyboard interrupt exception caught")
        
    cv2.destroyAllWindows()
    out_video.release()
    cam.release()
    if count:
        print("avg fps[%f] avg net_fps[%f]"%(t_fps_time / count, t_netfps_time / count))    
