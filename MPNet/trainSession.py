import time
import os.path as osp
import os
import shutil
import logging
from util_func import *
from vgg_2 import *
from vgg_fusion import *

def trainInitial(train_data, test_data, model_name, save_path):
    """
    Code for training the data
    :param train_data: Training data csv file
    :param test_data: Testing data csv file
    :param model_name: Name of the model to be saved
    :param save_path: Path of the model to be saved
    :return: None
    """
    # Initialization
    acc = 0
    step = 1
    index = 1
    process = 1
    roc_max = 0
    lr_index = 0
    dropout = 0.5
    batch_size = 256
    max_train_iters = 2000000
    save_dir = save_path
    net_name = model_name

    # Back up code
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') + net_name
    model_save_dir = osp.join(save_dir, time_stamp)
    if not osp.exists(model_save_dir):
        os.mkdir(model_save_dir)
    log_fn = osp.join(save_dir, time_stamp, 'log.txt')
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', filename=log_fn, level=logging.DEBUG)

    # Load data
    train_file = train_data
    train_hist, train_label = data_load2(train_file)
    x_mean = np.mean(train_hist, axis=0)
    x_std = np.std(train_hist, axis=0)
    np.savez('x_pra.npz', x_mean=x_mean, x_std=x_std)
    test_file = test_data
    test_hist, test_label = data_load2(test_file)
    test_hist = (test_hist - x_mean) / x_std
    mean = tf.Variable(x_mean, trainable=False)
    std = tf.Variable(x_std, trainable=False)

    # build network
    with tf.device('/gpu:0'):
        learning_rate = tf.Variable(0.6 , trainable=False)  # 0.000005, trainable=False)
        x = tf.placeholder(tf.float32, [None, 256, 1, 1])
        y = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)
    pred_out = model(x, keep_prob, training=True)[0]
    pred_soft = tf.nn.softmax(pred_out)
    [cost, accuracy] = loss(pred_out, y)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # Session configurationr
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'

    # Create session
    with tf.Session(config=config) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(model_save_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        while step < max_train_iters and process == 1:
            batch_xs = train_hist[(index - 1) * batch_size: index * batch_size, :, :, :]
            batch_ys = train_label[(index - 1) * batch_size: index * batch_size]
            batch_xs = (batch_xs - mean.eval()) / std.eval()
            [op, acc_train, cost_train, lr] = sess.run([optimizer, accuracy, cost, learning_rate],
                                                       feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout })
            print("step:{}\t acc:{:.4f}\t loss:{:.4f}\t valid_acc:{:.4f} learning_rate:{:.12f}"
                  .format(step, acc_train, np.sum(cost_train), acc, lr))

            if step % 1000 == 0:
                summ = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                tf.summary.scalar('loss', np.sum(cost_train))
                writer.add_summary(summ, step)

            if index < train_hist.shape[0] / batch_size:
                index = index + 1
            else:
                index = 1
                index_list = np.random.permutation(np.shape(train_label)[0])
                train_hist = train_hist[index_list]
                train_label = train_label[index_list]
                acc = 0
                pred_prob = np.ones(((np.shape(test_label)[0] / batch_size) * batch_size, 2))
                for i in range((np.shape(test_hist)[0] / batch_size)):
                    pred_prob[i * batch_size: (i + 1) * batch_size], acc_test = sess.run([pred_soft, accuracy],
                                          feed_dict={x: test_hist[i * batch_size: (i + 1) * batch_size, :, :, :],
                                                     y: test_label[i * batch_size: (i + 1) * batch_size, :], keep_prob: dropout})
                    acc = acc + np.array(acc_test)
                acc = acc / i
                label = test_label[0:np.shape(test_label)[0] / batch_size * batch_size]
                roc_auc = roc_create(label[:,0],pred_prob[:,0])[2]
                logging.info(
                    "step:{}\t train_acc:{} valid_acc:{} cost:{} lr:{}".format(step, acc_train, acc, np.sum(cost_train), lr))
                [roc_max, lr_index, process] = early_stoping(sess, roc_auc, roc_max, lr_index, learning_rate, process)
                saver.save(sess, model_save_dir + '/' + net_name + '_', global_step=step)

            step = step + 1


def trainFinetune(train_data, test_data, model_name, save_path):
    """
    Code for training the fusion model
    :param train_data: Training data csv file
    :param test_data: Testing data csv file
    :param model_name: Name of the model to be saved
    :param save_path: Path of the model to be saved
    :return: None
    """

    # Initialization
    acc = 0
    step = 1
    index = 1
    roc_max = 0
    process = 1
    lr_index = 0
    dropout = 0.5
    batch_size = 256
    max_train_iters = 2000000
    save_dir = save_path
    net_name = model_name
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') + net_name
    model_save_dir = osp.join(save_dir, time_stamp)

    # Backup Code
    if not osp.exists(model_save_dir):
        os.mkdir(model_save_dir)
    log_fn = osp.join(save_dir, time_stamp, 'log.txt')
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', filename=log_fn, level=logging.DEBUG)

    # Load Data
    train_file = train_data
    train_hist, train_label = data_load2(train_file)
    x_mean = np.mean(train_hist, axis=0)
    x_std = np.std(train_hist, axis=0)
    np.savez('x_pra.npz', x_mean=x_mean, x_std=x_std)
    test_file = test_data
    test_hist, test_label = data_load2(test_file)
    test_hist = (test_hist - x_mean) / x_std
    mean = tf.Variable(x_mean, trainable=False)
    std = tf.Variable(x_std, trainable=False)

    # build network
    with tf.device('/gpu:0'):
        learning_rate = tf.Variable(0.4, trainable=False)
        x = tf.placeholder(tf.float32, [None, 256, 1, 1])
        y = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)
    data_dict = np.load('vgg_fusion.npz')
    pred_out, weight, basis = model_fusion(x, keep_prob, data_dict, training=False)
    pred_soft = tf.nn.softmax(pred_out)
    [cost, accuracy] = loss(pred_out, y)
    reg = tf.nn.l2_loss(weight) + tf.nn.l2_loss(basis)
    cost = tf.reduce_sum(cost + reg*0.01)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # Session configurationr
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'

    # Create Session
    with tf.Session(config=config) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(model_save_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        while step < max_train_iters and process == 1:
            batch_xs = train_hist[(index - 1) * batch_size: index * batch_size, :, :, :]
            batch_ys = train_label[(index - 1) * batch_size: index * batch_size]
            batch_xs = (batch_xs - mean.eval()) / std.eval()
            [op, acc_train, cost_train, lr] = sess.run([optimizer, accuracy, cost, learning_rate],
                                                       feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout })
            print("step:{}\t acc:{:.4f}\t loss:{:.4f}\t valid_acc:{:.4f} learning_rate:{:.12f}".format(step,
                                         acc_train, np.sum(cost_train), acc, lr))

            if step % 1000 == 0:
                summ = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                tf.summary.scalar('loss', np.sum(cost_train))
                writer.add_summary(summ, step)

            if index < train_hist.shape[0] / batch_size:
                index = index + 1
            else:
                index = 1
                index_list = np.random.permutation(np.shape(train_label)[0])
                train_hist = train_hist[index_list]
                train_label = train_label[index_list]
                acc = 0
                pred_prob = np.ones(((np.shape(test_label)[0] / batch_size) * batch_size, 2))
                for i in range((np.shape(test_hist)[0] / batch_size)):
                    pred_prob[i * batch_size: (i + 1) * batch_size], acc_test = sess.run([pred_soft, accuracy],
                              feed_dict={x: test_hist[i * batch_size: (i + 1) * batch_size, :, :, :],
                                         y: test_label[i * batch_size: (i + 1) * batch_size, :], keep_prob: dropout})
                    acc = acc + np.array(acc_test)
                acc = acc / i
                label = test_label[0:np.shape(test_label)[0] / batch_size * batch_size]
                roc_auc = roc_create(label[:,0], pred_prob[:,0])[2]
                logging.info(
                    "step:{}\t train_acc:{} valid_acc:{} cost:{} lr:{}".format(step, acc_train, acc, np.sum(cost_train), lr))
                [roc_max, lr_index, process] = early_stoping(sess, roc_auc, roc_max, lr_index, learning_rate, process)
                saver.save(sess, model_save_dir + '/' + net_name + '_', global_step=step)

            step = step + 1