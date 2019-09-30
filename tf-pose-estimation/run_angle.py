import argparse
import logging
import sys
import time
import math

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  return  math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi

def length_between_points(p0, p1):
    return math.hypot(p1[0]- p0[0], p1[1]-p0[1])

def human_cnt(humans):
    if humans is None:
        return 0
    return len(humans)

def get_angle_point(human, pos):
    pnts = []

    if pos == 'left_elbow':
        pos_list = (5,6,7)
    elif pos == 'left_hand':
        pos_list = (11,5,7)
    elif pos == 'left_knee':
        pos_list = (11,12,13)
    elif pos == 'left_ankle':
        pos_list = (5,11,13)
    elif pos == 'right_elbow':
        pos_list = (2,3,4)
    elif pos == 'right_hand':
        pos_list = (8,2,4)
    elif pos == 'right_knee':
        pos_list = (8,9,10)
    elif pos == 'right_ankle':
        pos_list = (2,8,10)
    else:
        logger.error('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if pos_list[i] not in human.body_parts.keys():
            logger.info('component [%d] incomplete', pos_list[i])
            return pnts
        p = human.body_parts[pos_list[i]]
        pnts.append((int(p.x * image_w + 0.5), int(p.y * image_h + 0.5)))
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
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/hong.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_v2_small',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    img_name = './images/' + args.model + '.png'

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        w = 432
        h = 368
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    logger.info('Image resized=(W:%d, H:%d)' % (w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    #test
    img = image.copy()
    #img = cv2.medianBlur(img,5)
    #img = cv2.bilateralFilter(img,6,25,25)
    cv2.imwrite("./images/prefilter.png", img)
    t = time.time()
    humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t
    human_num = human_cnt(humans)

    logger.info('inference image: %s in %.4f seconds. ==>find %d humans' % (args.image, elapsed, human_num))
    image_h, image_w = image.shape[:2]
    print('Image size W:%d H:%d'%(image_w, image_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    '''
    print('==== part count ==========')
    print(humans[0].part_count())
    print('======== key =========')
    for key in humans[0].body_parts.keys():
        print(key)
        print(humans[0].body_parts[key])
    '''

    parts = [part for idx, part in humans[0].body_parts.items()]
    for part in parts:
        print(part.get_part_name())
        print('No:%d X:%d Y:%d Score:%f'%( part.part_idx, int(part.x * image_w + 0.5), int(part.y * image_h + 0.5), part.score))
        cv2.putText(image,str(part.part_idx),  (int(part.x * image_w + 0.5), int(part.y * image_h + 0.5)), font, 0.5, (0,255,0), 2)
    #cv2.imwrite(img_name, image)
    #print('===== part=========')
    #print(dir(parts[0]))
    #print('==============')
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    cv2.imwrite(img_name, image)
    angle_left_hand(humans[0])
    angle_left_elbow(humans[0])
    angle_left_knee(humans[0])
    angle_left_ankle(humans[0])

    angle_right_hand(humans[0])
    angle_right_elbow(humans[0])
    angle_right_knee(humans[0])
    angle_right_ankle(humans[0])


