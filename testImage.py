import os
import argparse
from util_func import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagePath', type=str, default='test.jpg')
    parser.add_argument('--netName', type=str, default='VGG_Fusion')
    args = parser.parse_args()

    net_name = args.netName
    name = args.imagePath
    data = loadImage(name)

    # Choose the net to build
    if net_name == 'VGG_Fusion':
        from vgg_fusion import *

        model_path = '../icpr_model/vgg_fusion/vgg_fusion'
        ouput_dim = 2
    elif net_name == 'VGG_2':
        from vgg_2 import *

        model_path = '../icpr_model/vgg_2/vgg_2'
        ouput_dim = 2
    elif net_name == 'VGG_3':
        from vgg_3 import *
        model_path = '../icpr_model/vgg_3/vgg_3 '
        ouput_dim = 3
    elif net_name == 'VGG_3D':
        from vgg_3D import *

        model_path = '../icpr_model/vgg_3d/vgg_3d'
        ouput_dim = 3
    elif net_name == 'VGG_2D':
        from vgg_2D import *

        model_path = '../icpr_model/vgg_2d/vgg_2d'
        ouput_dim = 2
    elif net_name == 'VGG_2_dp':
        from vggb_b import *
        model_path = '../icpr_model/vgg_2_dp/vgg_2dp'
        ouput_dim = 2

    # Normalization for the data
    data_dict0 = np.load('x_pra.npz')
    xx_mean = data_dict0['x_mean']
    xx_std = data_dict0['x_std']

    # Initialization for the tensor graph
    mean = tf.Variable(xx_mean, trainable=False)
    std = tf.Variable(xx_std, trainable=False)
    learning_rate = tf.Variable(0.001, trainable=False)
    x = tf.placeholder(tf.float32, [None, 256, 1, 1])
    keep_prob = tf.placeholder(tf.float32)
    pred_out = model(x, keep_prob, training=False)[0]
    pred = tf.nn.softmax(pred_out)

    # Build the graph
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'

    # Create the session
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("model restored")
        bin = (data - mean.eval()) / (std.eval() + 0.00000001)
        score = np.array(sess.run(pred, {x: bin, keep_prob: 1}))
        print('The probability of the image being forged is ', score[0, 0])