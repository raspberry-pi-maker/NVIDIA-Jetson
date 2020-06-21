from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
import requests, sys, time, os
from PIL import Image, ImageDraw
import argparse



COLORS = [(0, 45, 74), (85, 32, 98), (93, 69, 12), (49, 18, 55), (46, 67, 18), (30, 74, 93),
(218, 0, 0), (0, 218, 0), (0, 0, 218),(218, 218, 0), (0, 218, 218), (218, 0, 218),
(128, 0, 0), (0, 128, 0), (0, 0, 128),(128, 128, 0), (0, 128, 128)]

help = 'Base-Keypoint-RCNN-FPN'
help += ',keypoint_rcnn_R_101_FPN_3x'
help += ',keypoint_rcnn_R_50_FPN_1x'
help += ',keypoint_rcnn_R_50_FPN_3x'
help += ',keypoint_rcnn_X_101_32x8d_FPN_3x'


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'keypoint_rcnn_R_50_FPN_1x', help = help)
parser.add_argument('--file', type=str, default = '')
parser.add_argument('--size', type=str, default = '640X480', help = 'image inference size ex:320X240')

opt = parser.parse_args()

W, H = opt.size.split('X')
if opt.file == '':
    url = 'http://images.cocodataset.org/val2017/000000439715.jpg'
    img = Image.open(requests.get(url, stream=True).raw).resize((int(W),int(H)))
    im = np.asarray(img, dtype="uint8")
    height, width, channels = im.shape
    if channels == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    
else:
    im = cv2.imread(opt.file, cv2.IMREAD_COLOR)
    height, width, channels = im.shape
    

print('image W:%d H:%d'%(width, height))

network_model = 'COCO-Keypoints/' + opt.model + '.yaml'


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(network_model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(network_model)
predictor = DefaultPredictor(cfg)
kname = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names

im2 = im.copy()
font = cv2.FONT_HERSHEY_SIMPLEX  # normal size sans-serif font
outputs = predictor(im)
kpersons = outputs["instances"].pred_keypoints
#for kpoints in kperson[0]:
for kperson in kpersons:
    print('==== person ====')
    for i in range(0, len(kperson)):
        kpoints = kperson[i].cpu()
        x = kpoints[0] 
        y = kpoints[1]
        print('%-20s position  (%f, %f)'%(kname[i], x,y))
        cv2.circle(im2,  (int(x), int(y)), 10, color = COLORS[i], thickness=4)
        cv2.putText(im2, kname[i], (int(x) - 20, int(y) - 10), font, fontScale = 1, color = COLORS[i], thickness = 2)

cv2.imwrite("detectron2_PoseEstimation_%s_result.jpg"%(opt.model), im2)



