import os
import cv2
import time
import random
import shutil
import logging
import os.path as osp
from util_func import *
from netForImage import *
from skimage import exposure


def model_train(train_file, test_file, net_name, save_dir):
    """
    Code for training the data
    :param train_file: Training data csv file
    :param test_file: Testing data csv file
    :param net_name: Name of the model to be saved
    :param save_path: Path of the model to be saved
    :return: None
    """

    max_train_iters = 200000
    dropout = 0.5
    acc_max = 0
    lr_index = 0
    step = 1
    index = 1
    process = 1
    acc = 0
    batch_size = 32

    # Backup code
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') + net_name
    model_save_dir = osp.join(save_dir, time_stamp)
    if not osp.exists(model_save_dir):
        os.mkdir(model_save_dir)
    log_fn = osp.join(save_dir, time_stamp, 'log.txt')
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', filename=log_fn, level=logging.DEBUG)

    # Load Data
    train_hist, train_label = create_data(train_file)
    train_hist = np.array(train_hist)
    train_label = np.array(train_label)
    x_mean = np.mean(train_hist, axis=0)
    x_std = np.std(train_hist, axis=0)
    test_hist, test_label = create_data(test_file)
    test_hist = np.array(test_hist)
    test_label = np.array(test_label)
    test_hist = (test_hist - x_mean) / x_std
    test_hist = np.array(test_hist)
    test_hist = test_hist.reshape(np.shape(test_hist)[0], 224, 224, 3)

    # build network
    mean = tf.Variable(x_mean, trainable=False)
    std = tf.Variable(x_std, trainable=False)
    with tf.device('/gpu:0'):
        learning_rate = tf.Variable(0.2, trainable=False)  # 0.000005, trainable=False)
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)
    data = np.load('vgg16_weights.npz')
    pred_out = net(x, keep_prob ,data)
    [cost, accuracy] = loss(pred_out, y)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # Session configurationr
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '1'

    # Create Session
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(model_save_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        print("model restored")

        while step < max_train_iters and process == 1:
            batch_xs = train_hist[(index - 1) * batch_size: index * batch_size]
            batch_ys = train_label[(index - 1) * batch_size: index * batch_size]
            ratio = np.sum(batch_ys[:, 0]) / len(batch_ys[:, 0])
            batch_xs = (batch_xs - mean.eval()) / std.eval()
            batch_xs = np.array(batch_xs)
            batch_xs = batch_xs.reshape(np.shape(batch_xs)[0], 224, 224, 3)

            [op, acc_train, cost_train, lr] = sess.run([optimizer, accuracy, cost, learning_rate],
                                                       feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout })
            print("step:{}\t acc:{:.4f}\t loss:{:.4f}\t valid_acc:{:.4f} learning_rate:{:.12f} ratio:{:.4f}".format(
                step, acc_train, np.sum(cost_train), acc, lr, ratio))

            if step % 1000 == 0:
                summ = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                tf.summary.scalar('loss', np.sum(cost_train))
                writer.add_summary(summ, step)

            if index < np.shape(train_hist)[0] / batch_size:
                index = index + 1
            else:
                index = 1
                acc = 0
                for i in range((np.shape(test_hist)[0] / batch_size)):
                    [acc_test] = sess.run([accuracy], feed_dict={x: test_hist[i * batch_size: (i + 1) * batch_size],
                                        y: test_label[i * batch_size: (i + 1) * batch_size], keep_prob: dropout})
                    acc = acc + np.array(acc_test)
                acc = acc / i
                logging.info("step:{}\t train_acc:{} valid_acc:{} cost:{} lr:{}".format(step, acc_train,
                                                                                        acc, np.sum(cost_train), lr))
                [acc_max, lr_index, process] = early_stoping(sess, acc, acc_max, lr_index, learning_rate, process)
                saver.save(sess, model_save_dir + '/' + net_name + '_', global_step=step)

            step = step + 1
