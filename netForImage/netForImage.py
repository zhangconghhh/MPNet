import numpy as np
import layers as L
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def testModel(net, keep_prob, training=True):

    # assuming 224x224x3 input_tensor
    # block 1 -- outputs 112x112x64
    net, w11, b11 = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64, training=training)
    net, w12, b12 = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64, training=training)
    net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    net, w21, b21 = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128, training=training)
    net, w22, b22 = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128, training=training)
    net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    net, w31, b31 = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256, training=training)
    net, w32, b32 = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256, training=training)
    net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512/
    net, w41, b41 = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512, training=training)
    net, w42, b42 = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512, training=training)
    net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    net, w51, b51 = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512, training=True)
    net, w52, b52 = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512, training=True)
    net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    net, w6, b6 = L.fully_connected(net, name="fc6", n_out=32, training=True)
    net = tf.nn.dropout(net, keep_prob)
    net, w7, b7 = L.fully_connected(net, name="fc7", n_out=2, training=True)

    return net


def trainModel(net, keep_prob, data, training=True):

    # assuming 224x224x3 input_tensor
    # block 1 -- outputs 112x112x64
    net, w11, b11 = L.conv_f(net, name="conv1_1", data=data, training=training)
    net, w12, b12 = L.conv_f(net, name="conv1_2", data=data, training=training)
    net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    net, w21, b21 = L.conv_f(net, name="conv2_1", data=data, training=training)
    net, w22, b22 = L.conv_f(net, name="conv2_2", data=data, training=training)
    net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    net, w31, b31 = L.conv_f(net, name="conv3_1", data=data, training=training)
    net, w32, b32 = L.conv_f(net, name="conv3_2", data=data,  training=training)
    net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512/
    net, w41, b41 = L.conv_f(net, name="conv4_1", data=data, training=training)
    net, w42, b42 = L.conv_f(net, name="conv4_2", data=data, training=training)
    net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    net, w51, b51 = L.conv_f(net, name="conv5_1", data=data, training=training)
    net, w52, b52 = L.conv_f(net, name="conv5_2", data=data, training=training)
    net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    net, w6, b6 = L.fully_connected(net, name="fc6", n_out=32, training=training)
    net = tf.nn.dropout(net, keep_prob)
    net, w7, b7 = L.fully_connected(net, name="fc7", n_out=2, training=training)

    return net
