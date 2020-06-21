from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg


network_model = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(network_model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(network_model)
cls = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
print('NetworkModel[%s] category:%d'%(network_model, len(cls)))
print(cls)

kname = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names
print('keypoint_names')
print(kname)

krule = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_connection_rules
print('keypoint_connection_rules length[%d]'%(len(krule)))
print(krule)

