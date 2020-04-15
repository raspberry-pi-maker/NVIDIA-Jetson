
import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')
parser.add_argument('--makeup', type=str, default=os.path.join('imgs', 'makeup', 'XMY-014.png'), help='path to the makeup image')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)

batch_size = 1
img_size = 256
no_makeup = cv2.resize(cv2.imread(args.no_makeup, cv2.IMREAD_COLOR), (img_size, img_size))
no_makeup = cv2.cvtColor(no_makeup, cv2.COLOR_BGR2RGB)

X_img = np.expand_dims(preprocess(no_makeup), 0)
makeup_image = args.makeup

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('models', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('models'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

makeup = cv2.resize(imread(makeup_image), (img_size, img_size))
Y_img = np.expand_dims(preprocess(makeup), 0)
Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
Xs_ = deprocess(Xs_)
img = cv2.cvtColor(Xs_[0], cv2.COLOR_RGB2BGR)
cv2.imwrite('makeup.jpg', img)


