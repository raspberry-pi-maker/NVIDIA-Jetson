import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import time
import tarfile
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tf_sess = None
graph_def = None
parser = argparse.ArgumentParser(description='object_detection using  tensorRT')
parser.add_argument('--trtmodel', type=str, required=True, help='target tensorRT optimized model path')
parser.add_argument('--image', type=str, required=True, help='inference image file path')
args = parser.parse_args()

 
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def load_graph():
  gf = tf.GraphDef()
  with tf.gfile.GFile(args.trtmodel, 'rb') as fid:
    gf.ParseFromString(fid.read())
  return  gf

def make_session(graph_def):
  global tf_sess
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  #tf_sess = tf.Session(config=tf_config, graph = graph_def)
  tf_sess = tf.Session(config=tf_config)
  tf.import_graph_def(graph_def, name='')

def run_inference_for_single_image2(image):
    global tf_sess

    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor' + ':0')
    tensor_dict = {}
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}

    #for key in [ 'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks' ]:
    for key in [ 'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    t = time.time()
    output_dict = tf_sess.run(tensor_dict, feed_dict={tf_input: image})
    elapsed = time.time() - t
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[ 'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict, elapsed

graph_def = load_graph()
make_session(graph_def)
print('===== Image open:%s ====='%(args.image))  
im = Image.open(args.image)
width, height = im.size 
#image = im.resize((int(width / 2), int(height / 2)))
image = im.copy()

# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.
image_np = load_image_into_numpy_array(image)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
image_np_expanded = np.expand_dims(image_np, axis=0)
# Actual detection.
#output_dict, elapsed = run_inference_for_single_image(image_np_expanded, graph_def)
output_dict, elapsed = run_inference_for_single_image2(image_np_expanded)

# Visualization of the results of a detection.
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)
fig = plt.figure(figsize=IMAGE_SIZE)
txt = 'FPS:%f'%(1.0 / elapsed) 
plt.text(10, 10, txt, fontsize=12)
plt.imshow(image_np)
name = os.path.splitext(args.image)[0] 
name = name + '_result_rt.png'
plt.savefig(name)
  
  
