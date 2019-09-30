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

'''
hnum: 0 based human index
pos : keypoint
'''
def get_keypoint(humans, hnum, pos):
    #check invalid human index
    if human_cnt(humans) <= hnum:
        return None

    #check invalid keypoint. human parts may not contain certain ketpoint
    if pos not in humans[hnum].body_parts.keys():
        return None

    part = humans[hnum].body_parts[pos]
    return part

'''
return keypoint posution (x, y) in the image
'''
def get_point_from_part(image, part):
    image_h, image_w = image.shape[:2]
    return (int(part.x * image_w + 0.5), int(part.y * image_h + 0.5))


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
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
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

    for i in range(18):
        part = get_keypoint(humans, 0, i)
        if part is None:
            continue
        pos =  get_point_from_part(image, part)
        print('No:%d Name[%s] X:%d Y:%d Score:%f'%( part.part_idx, part.get_part_name(),  pos[0] , pos[1] , part.score))
        cv2.putText(image,str(part.part_idx),  (pos[0] + 10, pos[1]), font, 0.5, (0,255,0), 2)

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
    cv2.imwrite(img_name, image)


