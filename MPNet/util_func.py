import csv
import pdb
import cv2
import numpy as np
import tensorflow as tf
from scipy import interp
from sklearn.metrics import roc_curve, auc


def loadImage(name):
    """
    Function for data loading
    :param name: Name of the image
    :return:  The data extracted from the data
    """
    im = cv2.imread(name)
    hist = np.histogram(im[:, :, 0], bins=256, range=(0, 256), normed=True)[0]
    print('Finish Data Loading')
    return np.reshape(hist, (1, 256, 1, 1))


def loss(logits, labels):
    """
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    correct_pred_num = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred_num, tf.float32))
    # summ params
    tf.add_to_collection('accuracy', accuracy)
    return cross_loss, accuracy


def early_stoping(sess, acc_te, acc_max, lr_index, learning_rate, process):
    """
    Criterion for early stopping
    :param acc_te: Accuracy for the last evaluation
    :param acc_max: Maximum accuracy
    :param lr_index: Index for deciding if to end the training process
    :param learning_rate: Learning rate of the current process
    :param process: Flag for the process, 1 for continue, 0 for stop
    :return: acc_max: Current maximum accuracy
             lr_index: Index for deciding if to end the training process
             process: Flag for the process, 1 for continue, 0 for stop
    """
    if acc_te > acc_max:
        acc_max = acc_te
        lr_index = 0
    else:
        lr_index = lr_index + 1
    if lr_index == 10:
        if learning_rate.eval() > 0.01 / 2 ** 512:
            lr_decay_op = learning_rate.assign(learning_rate / 2.0)
            sess.run(lr_decay_op)
            print("learning_rate_decay:{:.8f}".format(lr_decay_op.eval()))
            lr_index = 0
        else:
            process = 0
    return acc_max, lr_index, process


def roc_create(label, pos_score):
    """
    Given ground truth label and predicted positive class score, create ROC curve.
    :param label: array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.
    :param pos_score: array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    :return: fpr: the false positive
             tpr: the truth positive
             roc_auc: the AUC of the ROC curve
             mean_tpr: the ROC curve
    """
    [fpr, tpr, thresholds] = roc_curve(label, pos_score)
    mean_tpr = 0
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, mean_tpr


def data_load2(file_name):
    """
    extract train/test/valid and bin/label from the file
    :param file_name: file_name
    :return: bin:  N_samples * feature_bumber
            label: N_samples * number_class
    """
    data = []
    with open(file_name, 'r', newline = '') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    data = np.array(data)
    #np.random.shuffle(data)
    # print np.shape(data)
    # print data
    bin = data[:, 0:256]
    bin = bin.astype(np.float32)
    label = data[:, 256:258]
    label = label.astype(np.float32)
    for ith, i_label in enumerate(label):
        if np.array_equal(i_label, [0.0, 1.0]) or np.array_equal(i_label, [1.0, 0.0]):
            continue
        else:
            print('wrong load on {}th {}'.format(ith, i_label))
    bin = bin.reshape(np.shape(bin)[0], 256, 1, 1)
    return bin, label


def data_load_gamma(file_name):

    """
    extract train/test/valid and bin/label from the file
    :param file_name: file_name
    :return: bin:  N_samples * feature_bumber
            label: N_samples * number_class
    """
    data = []
    data_train = []
    with open(file_name, 'rb') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    data = np.array(data)
    np.random.shuffle(data)
    #pdb.set_trace()
    for i in range(np.shape(data)[0]):
        if data[i, 258] == '0' or data[i,258] == '1':
            data_train.append(data[i,:])
    data_train =np.array(data_train)
    bin = data_train[:, 0:256]
    bin = bin.astype(np.float32)
    label = data_train[:, 256:258]
    label = label.astype(np.float32)
    for ith, i_label in enumerate(label):
        if np.array_equal(i_label, [0.0, 1.0]) or np.array_equal(i_label, [1.0, 0.0]):
            continue
        else:
            print('wrong load on {}th {}'.format(ith, i_label))
    bin = bin.reshape(np.shape(bin)[0], 256, 1, 1)
    return bin, label


def data_load_his(file_name):

    """
    extract train/test/valid and bin/label from the file
    :param file_name: file_name
    :return: bin:  N_samples * feature_bumber
            label: N_samples * number_class
    """
    data = []
    data_train = []
    with open(file_name, 'rb') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    data = np.array(data)
    np.random.shuffle(data)
    for i in range(np.shape(data)[0]):
        if data[i, 258] == '0' or data[i,258] == '2':
            data_train.append(data[i,:])
    data_train =np.array(data_train)
    bin = data_train[:, 0:256]
    bin = bin.astype(np.float32)
    label = data_train[:, 256:258]
    label = label.astype(np.float32)
    for ith, i_label in enumerate(label):
        if np.array_equal(i_label, [0.0, 1.0]) or np.array_equal(i_label, [1.0, 0.0]):
            continue
        else:
            print('wrong load on {}th {}'.format(ith, i_label))
    bin = bin.reshape(np.shape(bin)[0], 256, 1, 1)
    return bin, label