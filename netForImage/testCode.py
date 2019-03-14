from netForImage import *
import os
from util_func import *
from skimage import exposure
import matplotlib.pylab as plt
import pdb
import cv2
import scipy.io as scio
import random



model_path = '../icpr_model/image_model/image_model'
flie_dir = '../histogram/test.jpg'


# Load Data
bin = create_data_image(flie_dir)
# pdb.set_trace()


#Build training graph
mean = tf.Variable(np.zeros((224*224, 3)), trainable=False)
std = tf.Variable(np.zeros((224*224, 3)), trainable=False)
learning_rate = tf.Variable(0.001, trainable=False)
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
pred_out = testModel(x, keep_prob, training=True)
pred = tf.nn.softmax(pred_out)
cross = loss(pred, y)[0]
accuracy = loss(pred, y)[1]

# Configuration
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
saver = tf.train.Saver()

# Create Session
with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)
    print("model restored")
    bin = (bin - mean.eval()) / (std.eval() + 0.00000001)
    bin = bin.reshape((1, 224, 224, 3))
    score = sess.run(pred, {x: bin, keep_prob: 1})
    print('The probability of the image being forged is ', score[0, 0])