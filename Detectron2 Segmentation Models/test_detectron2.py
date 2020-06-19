from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
import requests, sys, time, os
from PIL import Image

url = 'http://images.cocodataset.org/val2017/000000439715.jpg'

img = Image.open(requests.get(url, stream=True).raw)
im = np.asarray(img, dtype="uint8")
height, width, channels = im.shape
if channels == 3:
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
else:
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

print('image W:%d H:%d'%(width, height))

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


for i in range (5):
    fps_time  = time.perf_counter()
    outputs = predictor(im)

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))

    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes)


    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
cv2.imwrite("detectron2_result.jpg", out.get_image()[:, :, ::-1])
cv2.imwrite("./source_image.jpg", im)
