import argparse
import time
import math
import cv2
import numpy as np
from pose_engine import PoseEngine


def main():
    parser = argparse.ArgumentParser(description='PoseNet')
    parser.add_argument('--model', type=str, default='mobilenet')
    args = parser.parse_args()
    
    if args.model == 'mobilenet':
        model = 'models/mobilenet/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite'
    else:
        model = 'models/resnet/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite'
        
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()        
    inference_size = (input_shape[2], input_shape[1])
    print(inference_size)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('./output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), inference_size)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if cap is None:
        print("Camera Open Error")
        sys.exit(0)
    
    count = 0
    total_ftp = 0.0
    fps_cnt = 0
    while cap.isOpened() and count < 60:
        ret_val, img = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break    
        print('frame read')
        img = cv2.resize(img, inference_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s_time = time.time()
        poses, inference_time = engine.DetectPosesInImage(rgb)
        fps = 1.0 / (time.time() - s_time)
        total_ftp += fps
        fps_cnt += 1
        for pose in poses:
            print('\nPose Score: %f  FPS:%f'%(pose.score, fps))
            if pose.score < 0.3: continue
            for label, keypoint in pose.keypoints.items():
                print(' %-20s x=%-4d y=%-4d score=%.1f' %(label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
                p1 = (keypoint.yx[1], keypoint.yx[0])
                p2 = (keypoint.yx[1] + 5, keypoint.yx[0] + 5)
                cv2.circle(img, (keypoint.yx[1], keypoint.yx[0]), 2, (0,255,0), -1)

        out_video.write(img)                
        count += 1
    if fps_cnt > 0:   
        print('Model[%s] Avg FPS: %f'%(args.model, total_ftp / fps_cnt))
    cv2.destroyAllWindows()        
    cap.release()    
    out_video.release()
    
if __name__ == '__main__':
    main()


