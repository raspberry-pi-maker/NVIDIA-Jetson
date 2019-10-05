import argparse
import logging
import sys
import time
import math
import cv2
import numpy as np
from openpose import pyopenpose as op

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  if a * b == 0:
      return -1.0 
  return  math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi

def length_between_points(p0, p1):
    return math.hypot(p1[0]- p0[0], p1[1]-p0[1])


def get_angle_point(human, pos):
    pnts = []

    if pos == 'left_elbow':
        pos_list = (5,6,7)
    elif pos == 'left_hand':
        pos_list = (1,5,7)
    elif pos == 'left_knee':
        pos_list = (12,13,14)
    elif pos == 'left_ankle':
        pos_list = (5,12,14)
    elif pos == 'right_elbow':
        pos_list = (2,3,4)
    elif pos == 'right_hand':
        pos_list = (1,2,4)
    elif pos == 'right_knee':
        pos_list = (9,10,11)
    elif pos == 'right_ankle':
        pos_list = (2,9,11)
    else:
        logger.error('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if human[pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete'%(pos_list[i]))
            return pnts

        pnts.append((int( human[pos_list[i]][0]), int( human[pos_list[i]][1])))
    return pnts



def angle_left_hand(human):
    pnts = get_angle_point(human, 'left_hand')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return -1

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left hand angle:%f'%(angle))
    return angle


def angle_left_elbow(human):
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left elbow angle:%f'%(angle))
    return angle

def angle_left_knee(human):
    pnts = get_angle_point(human, 'left_knee')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left knee angle:%f'%(angle))
    return angle

def angle_left_ankle(human):
    pnts = get_angle_point(human, 'left_ankle')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left ankle angle:%f'%(angle))
    return angle

def angle_right_hand(human):
    pnts = get_angle_point(human, 'right_hand')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right hand angle:%f'%(angle))
    return angle


def angle_right_elbow(human):
    pnts = get_angle_point(human, 'right_elbow')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right elbow angle:%f'%(angle))
    return angle

def angle_right_knee(human):
    pnts = get_angle_point(human, 'right_knee')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right knee angle:%f'%(angle))
    return angle

def angle_right_ankle(human):
    pnts = get_angle_point(human, 'right_ankle')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right ankle angle:%f'%(angle))
    return angle



logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openpose pos estimation run')
    parser.add_argument('--image', type=str, default='/usr/local/src/openpose/examples/media/COCO_val2014_000000000294.jpg')
    args = parser.parse_args()

    fps_time = 0

    params = dict()
    params["model_folder"] = "../../models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    #imagePath = '/usr/local/src/openpose/examples/media/COCO_val2014_000000000474.jpg'
    #imagePath = '/usr/local/src/openpose/examples/media/COCO_val2014_000000000536.jpg'
    imagePath = args.image

    print("OpenPose start")
    dst = cv2.imread(imagePath)
    img_name = '/tmp/openpose_keypoint.png'

    #dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
    #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", dst)
    #continue
    datum = op.Datum()
    datum.cvInputData = dst
    opWrapper.emplaceAndPop([datum])
    newImage = datum.cvOutputData[:, :, :]
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", newImage)
    human_count = len(datum.poseKeypoints)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(human_count):
        for j in range(25):
            if datum.poseKeypoints[i][j][2] > 0.01:
                cv2.putText(newImage,str(j),  ( int(datum.poseKeypoints[i][j][0]) + 10,  int(datum.poseKeypoints[i][j][1])), font, 0.5, (0,255,0), 2) 
        print(datum.poseKeypoints[i])

    cv2.imwrite(img_name, newImage)
    cv2.destroyAllWindows()        

    for i in range(human_count):
        print('=================================')
        angle_left_hand(datum.poseKeypoints[i] )
        angle_left_elbow(datum.poseKeypoints[i] )
        angle_left_knee(datum.poseKeypoints[i] )
        angle_left_ankle(datum.poseKeypoints[i] )
        angle_right_hand(datum.poseKeypoints[i] )
        angle_right_elbow(datum.poseKeypoints[i] )
        angle_right_knee(datum.poseKeypoints[i] )
        angle_right_ankle(datum.poseKeypoints[i] )









