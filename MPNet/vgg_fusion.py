import numpy as np
import tensorflow as tf
import MPNet.layers as L


def model_fusion(net, keep_prob, data, training):

    # assuming 256x1x1 input_tensor
    # block 1 -- outputs 128x1x1
    net, w11, b11 = L.conv_f(net, name="conv1_1", data=data, training=training)
    net, w12, b12 = L.conv_f(net, name="conv1_2", data=data, training=training)
    net = L.pool(net, name="pool1", kh=2, kw=1, dw=2, dh=2)

    # block 2 -- outputs 64x1x1
    net, w21, b21 = L.conv_f(net, name="conv2_1", data=data, training=training)
    net, w22, b22 = L.conv_f(net, name="conv2_2", data=data, training=training)
    net = L.pool(net, name="pool2", kh=2, kw=1, dh=2, dw=2)

    # # block 3 -- outputs 32x1x1
    net, w31, b31 = L.conv_f(net, name="conv3_1", data=data, training=training)
    net, w32, b32 = L.conv_f(net, name="conv3_2", data=data, training=training)
    net = L.pool(net, name="pool3", kh=2, kw=1, dh=2, dw=2)

    # block 4 -- outputs 16x1x1
    net, w41, b41 = L.conv_f(net, name="conv4_1", data=data, training=training)
    net, w42, b42 = L.conv_f(net, name="conv4_2", data=data, training=training)
    net = L.pool(net, name="pool4", kh=2, kw=1, dh=2, dw=2)

    # block 5_g -- outputs 8x1x1
    # This branch is for the gamma correction
    net1, wg51, bg51 = L.conv_f(net, name="conv5_g_1", data=data, training=True)
    net1, wg52, bg52 = L.conv_f(net1, name="conv5_g_2", data=data, training=True)
    net1 = L.pool(net1, name="pool5", kh=2, kw=1, dw=2, dh=2)
    flattened_shape = np.prod([s.value for s in net1.get_shape()[1:]])
    net1 = tf.reshape(net1, [-1, flattened_shape], name="flatten")
    net1, wg6, bg6 = L.fully_connected_f(net1, name="fc6_g", data=data, training=True)
    net1 = tf.nn.dropout(net1, keep_prob)

    # block 5_h -- outputs 8x1x1
    # This branch is for the histogram equalization
    net2, wh51, bh51 = L.conv_f(net, name="conv5_h_1", data=data, training=True)
    net2, wh52, bh52 = L.conv_f(net2, name="conv5_h_2", data=data, training=True)
    net2 = L.pool(net2, name="pool5", kh=2, kw=1, dw=2, dh=2)
    flattened_shape = np.prod([s.value for s in net2.get_shape()[1:]])
    net2 = tf.reshape(net2, [-1, flattened_shape], name="flatten")
    net2, wh6, bh6 = L.fully_connected_f(net2, name="fc6_h", data=data, training=True)
    net2 = tf.nn.dropout(net2, keep_prob)

    # Concat the two branches
    net = tf.concat([net1, net2], 1)
    net, w7, b7 = L.fully_connected(net, name="fc7", n_out=2, training=True)

    return net, w7, b7

def model(net, keep_prob, training):

    # assuming 256x1x1 input_tensor
    # block 1 -- outputs 128x1x1
    net, w11, b11 = L.conv(net, name="conv1_1", kh=3, kw=1, n_out=64, training=training)
    net, w12, b12 = L.conv(net, name="conv1_2", kh=3, kw=1, n_out=64, training=training)
    net = L.pool(net, name="pool1", kh=2, kw=1, dw=2, dh=2)

    # block 2 -- outputs 64x1x1
    net, w21, b21 = L.conv(net, name="conv2_1", kh=3, kw=1, n_out=128, training=training)
    net, w22, b22 = L.conv(net, name="conv2_2", kh=3, kw=1, n_out=128, training=training)
    net = L.pool(net, name="pool2", kh=2, kw=1, dh=2, dw=2)

    # # block 3 -- outputs 32x1x1
    net, w31, b31 = L.conv(net, name="conv3_1", kh=3, kw=1, n_out=256, training=training)
    net, w32, b32 = L.conv(net, name="conv3_2", kh=3, kw=1, n_out=256, training=training)
    net = L.pool(net, name="pool3", kh=2, kw=1, dh=2, dw=2)

    # block 4 -- outputs 16x1x1
    net, w41, b41 = L.conv(net, name="conv4_1", kh=3, kw=1, n_out=512, training=training)
    net, w42, b42 = L.conv(net, name="conv4_2", kh=3, kw=1, n_out=512, training=training)
    net = L.pool(net, name="pool4", kh=2, kw=1, dh=2, dw=2)

    # block 5_g -- outputs 8x1x1
    # This branch is for the gamma correction
    net1, wg51, bg51 = L.conv(net, name="conv5_g_1", kh=3, kw=1, n_out=512, training=training)
    net1, wg52, bg52 = L.conv(net1, name="conv5_g_2", kh=3, kw=1, n_out=512, training=training)
    net1 = L.pool(net1, name="pool5", kh=2, kw=1, dw=2, dh=2)
    flattened_shape = np.prod([s.value for s in net1.get_shape()[1:]])
    net1 = tf.reshape(net1, [-1, flattened_shape], name="flatten")
    net1, wg6, bg6 = L.fully_connected(net1, name="fc6_g", n_out=32, training=training)
    net1 = tf.nn.dropout(net1, keep_prob)

    # block 5_h -- outputs 8x1x1
    # This branch is for the histogram equalization
    net2, wh51, bh51 = L.conv(net, name="conv5_h_1", kh=3, kw=1, n_out=512, training=training)
    net2, wh52, bh52 = L.conv(net2, name="conv5_h_2", kh=3, kw=1, n_out=512, training=training)
    net2 = L.pool(net2, name="pool5", kh=2, kw=1, dw=2, dh=2)
    flattened_shape = np.prod([s.value for s in net2.get_shape()[1:]])
    net2 = tf.reshape(net2, [-1, flattened_shape], name="flatten")
    net2, wh6, bh6 = L.fully_connected(net2, name="fc6_h", n_out=32, training=training)
    net2 = tf.nn.dropout(net2, keep_prob)

    net = tf.concat([net1, net2], 1)
    net, w7, b7 = L.fully_connected(net, name="fc7", n_out=2, training=True)

    # Concat the two branches
    return net, w11, b11, w12, b12, w21, b21, w22, b22, w31, b31, w32, b32, w41, b41, w42, b42, wg51, bg51, wg52, bg52, \
           wg6, bg6, wh51, bh51, wh52, bh52, wh6, bh6, w7, b7

