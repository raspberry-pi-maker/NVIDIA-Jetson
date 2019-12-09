import argparse
import sys, os
import time
import tensorflow as tf

ver=tf.__version__.split(".")
if(int(ver[0]) == 1 and int(ver[1]) <= 13):
#if tensorflow vereion <= 1.13.1 use this module
    print('tf Version <= 1.13')
    import tensorflow.contrib.tensorrt as trt
else:
#if tensorflow vereion > 1.13.1 use this module instead
    print('tf Version > 1.13')
    from tensorflow.python.compiler.tensorrt import trt_convert as trt



def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


parser = argparse.ArgumentParser(description='tf network model conversion to tensorrt')
parser.add_argument('--tfmodel', type=str, 
                        help='source tensorflow frozen model (pb file)')
parser.add_argument('--trtmodel', type=str, 
                        help='target tensorRT optimized model path')
parser.add_argument('--outputs', type=str, 
                        help="output string of tf model's last node optimized model path")
parser.add_argument('--precision', type=str, default='FP16',
                        help="FP16, FP32, INT16")
parser.add_argument('--max_batch_size', type=int, default=1,
                        help="batch size , default :1")
parser.add_argument('--max_workspace_size_bytes', type=int, default=3,
                        help="max_workspace_size(GB) , default :3")
args = parser.parse_args()

frozen_name = args.tfmodel
frozen_graph = get_frozen_graph(frozen_name)
print('=======Frozen Name:%s======='%(frozen_name));
# convert (optimize) frozen model to TensorRT model
your_outputs = args.outputs.split(',')
for item in your_outputs:
    print('=======outputs:=======' + item );

start = time.time()
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    is_dynamic_op=True,
    #minimum_segment_size=3,
    minimum_segment_size=50,
    maximum_cached_engines=int(1e3),
    max_batch_size=args.max_batch_size,# specify your max batch size
    max_workspace_size_bytes=args.max_workspace_size_bytes*(10**9),# specify the max workspace (2GB)
    #max_workspace_size_bytes= 1 << 25,# specify the max workspace (2GB)
    # precision, can be "FP32" (32 floating point precision) or "FP16"
    precision_mode=args.precision
    )

elapsed = time.time() - start
print('Tensorflow model => TensorRT model takes : %f'%(elapsed))

#write the TensorRT model to be used later for inference
rt_name = args.trtmodel
with tf.gfile.FastGFile(rt_name , 'wb') as f:
    f.write(trt_graph.SerializeToString())

