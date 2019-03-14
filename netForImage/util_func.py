import csv
import cv2
import pdb
import numpy as np
import tensorflow as tf
from scipy import interp
from sklearn.metrics import roc_curve, auc


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
    cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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
    if lr_index == 5:
        if learning_rate.eval() > 0.01 / 2 ** 256:
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


def create_data(img_dir):
    """
    Create the data
    :param img_dir: The direction for the raw image
    :return: training data and label
    """
    file_name = os.listdir(img_dir)
    gamma_value = np.arange(0.4, 2.1, 0.1)
    num = len(gamma_value) - 1
    np.random.shuffle(file_name)
    data = []
    label = []

    for i in range(len(file_name)):
        print(i)
        im = cv2.imread(str(img_dir) + str(file_name[i]))
        im = cv2.resize(im, (256, 256))
        w1 = random.randint(0, 28)
        d1 = random.randint(0, 28)
        im = im[w1: 224 + w1, d1: 224 + d1]
        data.append(im.reshape(224*224, 3))
        label.append([0, 1])
        Num_pre = random.randint(1, 2)
        if Num_pre == 1:
            index = random.randint(0, num)
            im_eh = exposure.adjust_gamma(im, gamma_value[index])
        else:
            im_eh = np.zeros(np.shape(im))
            for i in range(3):
                im_eh[:, :, i] = cv2.equalizeHist(im[:, :, i])
        cv2.imwrite('tmp.jpg', im_eh)
        im_eh = cv2.imread('tmp.jpg')
        data.append(im_eh.reshape(224*224, 3))
        label.append([1, 0])

    return data, label


def create_data_image(img_dir):
    """
    Create the data for image input
    :param img_dir: The direction for the raw image
    :return: training data and label
    """
    im = cv2.imread(img_dir)
    im.resize(1, 224 * 224, 3)
    return im


def fp_tp(pred, label, thred=0.5):
    """
    Given a pred score vector and a groundtruth vector, return FP, TP, FN, TN
    :param pred: array, shape: N_samples*1, >0.5 for positive and <0.5 for negtive
    :param label: array, shape: N_samples*1, 1 for positive and 0 for negtive
    :return: FP, FN, TP, TN, N, P
             FP: False Positive sample number
             FN: False Negtive sample number
             TN: True Negtive smaple number
             TP: True Positive sample number
             N: Negtive sample number in ground truth
             P: Positive sample number in ground truth
    """

    FP, FN, TP, TN, N, P = 0, 0, 0, 0, 0, 0

    for ith, gt in enumerate(label):

        if gt == 1:
            P += 1
            if pred[ith] > thred:  # 11
                TP += 1
            elif pred[ith] <= thred:  # 10
                FN += 1
        elif gt == 0:
            N += 1
            if pred[ith] > thred:  # 01
                FP += 1
            elif pred[ith] <= thred:  # 00
                TN += 1

    return FP, FN, TP, TN, N, P