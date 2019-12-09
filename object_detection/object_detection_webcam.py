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
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tf_sess = None
parser = argparse.ArgumentParser(description='object_detection using  tensorRT')
parser.add_argument('--trtmodel', type=str, required=True, help='target tensorRT optimized model path')
args = parser.parse_args()

graph_def = tf.GraphDef()
with tf.gfile.GFile(args.trtmodel, 'rb') as fid:
  graph_def.ParseFromString(fid.read())

PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  return image    
#   (im_height, im_width) = image.shape
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)


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
    global tf_sess, graph_def

    # tf.import_graph_def(graph_def, name='')
    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor' + ':0')
    # tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    # tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    # tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    # tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0
tfps = 0.0
if cap is None:
    print("Camera Open Error")
    sys.exit(0)

while cap.isOpened() and count < 500:
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break
    image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
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
    fps = 1.0 / elapsed
    tfps += fps
    print("FPS:%f"%(fps))
    cv2.putText(image_np , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Object Detection", image_np)
    out_video.write(image_np)
    count += 1

print("AVG FPS:%f"%(tfps / 500.0))
  
cv2.destroyAllWindows()  
out_video.release()
cap.release()