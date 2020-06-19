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


COLORS = [(0, 45, 74, 127), (85, 32, 98, 127), (93, 69, 12, 127),
          (49, 18, 55, 127), (46, 67, 18, 127), (30, 74, 93, 127)]
LINE_COLORS = [(0, 45, 74, 255), (85, 32, 98, 255), (93, 69, 12, 255),
          (49, 18, 55, 255), (46, 67, 18, 255), (30, 74, 93, 255)]

help = 'mask_rcnn_R_101_C4_3x, mask_rcnn_R_101_DC5_3x, mask_rcnn_R_101_FPN_3x'
help += ',mask_rcnn_R_50_C4_1x'
help += ',mask_rcnn_R_50_C4_3x'
help += ',mask_rcnn_R_50_DC5_1x'
help += ',mask_rcnn_R_50_DC5_3x'
help += ',mask_rcnn_R_50_FPN_1x'
help += ',mask_rcnn_R_50_FPN_1x_giou'
help += ',mask_rcnn_R_50_FPN_3x'
help += ',mask_rcnn_X_101_32x8d_FPN_3x'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'mask_rcnn_R_50_FPN_3x', help = help)
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

network_model = 'COCO-InstanceSegmentation/' + opt.model + '.yaml'


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(network_model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(network_model)
predictor = DefaultPredictor(cfg)


fps_time  = time.perf_counter()
outputs = predictor(im)
fps = 1.0 / (time.perf_counter() - fps_time)
print('===== pred_boxes =====')
print(outputs["instances"].pred_boxes)

print('===== scores =====')
print(outputs["instances"].scores)

print('===== pred_classes =====')
print(outputs["instances"].pred_classes)

print('===== pred_masks len=====')
print(np.shape(outputs["instances"].pred_masks))

print("Net FPS: %f" % (fps))


'''
I'm going to use PIL draw, because it's very easy to draw alpha rectangle
'''

im2 = Image.fromarray(np.uint8(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
drw = ImageDraw.Draw(im2, 'RGBA')
pred_boxes = outputs["instances"].pred_boxes
classes = outputs["instances"].pred_classes

color_index = 0
for cs, box in zip(classes, pred_boxes):
    box = box.cpu()
    cs = cs.cpu()
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    
    color = COLORS[color_index % len(COLORS)]
    linecolor = LINE_COLORS[color_index % len(LINE_COLORS)]
    color_index += 1            
    drw.rectangle([x0, y0, x1, y1], outline = linecolor, width=5)
    drw.rectangle([x0, y0, x1, y1], fill = color, width=5)

im2.save("detectron2_box_%s_result.jpg"%(opt.model))       

