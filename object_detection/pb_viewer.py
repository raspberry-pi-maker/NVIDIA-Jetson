import argparse
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

parser = argparse.ArgumentParser(description='tf-model-conversion to TensorRT')
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
args = parser.parse_args()


with tf.Session() as sess:
    model_filename = args.model_dir
    with tf.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR=args.log_dir
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
