import argparse
import sys, os
import time
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


parser = argparse.ArgumentParser(description='tf network model conversion to tensorrt')
parser.add_argument('--model', type=str, default='mobilenet_v2_small',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
args = parser.parse_args()

model_dir = 'models/graph/'
frozen_name = model_dir + args.model + '/graph_opt.pb'
frozen_graph = get_frozen_graph(frozen_name)
print('=======Frozen Name:%s======='%(frozen_name));
# convert (optimize) frozen model to TensorRT model
your_outputs = ["Openpose/concat_stage7"]

start = time.time()
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    is_dynamic_op=True,
    minimum_segment_size=3,
    maximum_cached_engines=int(1e3),
    max_batch_size=1,# specify your max batch size
    max_workspace_size_bytes=2*(10**9),# specify the max workspace
    precision_mode="FP16") # precision, can be "FP32" (32 floating point precision) or "FP16"

elapsed = time.time() - start
print('Tensorflow model => TensorRT model takes : %f'%(elapsed));

#write the TensorRT model to be used later for inference
rt_name = model_dir + args.model + '/graph_opt_rt.pb' 
with tf.gfile.FastGFile(rt_name , 'wb') as f:
    f.write(trt_graph.SerializeToString())

