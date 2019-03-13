import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def _activation_summary(x, name):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tf.summary.histogram(name + '/activations', x)
  tf.summary.scalar(name + '/sparsity',tf.nn.zero_fraction(x))


def conv(input_tensor, name, kw, kh, n_out, training, dw=1, dh=1, activation_fn=tf.nn.relu):
    """
    Definition of the convolutional layer
    :param input_tensor: Input layer with the shape of [batchSiz, width, height, channel]
    :param name: Name of this layer
    :param kw: Width of the convolutional layer
    :param kh: Height of the convolutional layer
    :param n_out: Channel of the output tensor
    :param training: Flag for training
    :param dw: Stride size for width
    :param dh: Stride size for height
    :param activation_fn: Active function, default set as ReLu
    :return: Output of the layer, weights and biases
    """
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer(), trainable=training)
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0), trainable=training)
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        tf.summary.histogram(name + '/weight', weights)
        tf.summary.histogram(name + '/biases', biases)
        _activation_summary(conv, name)
        return activation, weights, biases


def fully_connected(input_tensor, name, n_out, training, activation_fn=tf.nn.relu):
    """
    Definition of the fully connected layer
    :param input_tensor: Input layer with the shape of [batchSiz, width, height, channel]
    :param name: Name of this layer
    :param n_out: Channel of the output tensor
    :param training: Flag for training
    :param activation_fn: Active function, default set as ReLu
    :return: Output of the layer, weights and biases
    """
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer(), trainable=training)
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0), trainable=training)
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        tf.summary.histogram(name + '/weight', weights)
        tf.summary.histogram(name + '/biases', biases)
        logits = activation_fn(logits)
        _activation_summary(logits, name)
        return logits, weights, biases


def conv_f(input_tensor, name, data, training, dw=1, dh=1, activation_fn=tf.nn.relu):
    """
    Definition of the convolutional layer for finetune
    :param input_tensor: Input layer with the shape of [batchSiz, width, height, channel]
    :param name: Name of this layer
    :param data: Data used for initialization
    :param training: Flag for training
    :param dw: Stride size for width
    :param dh: Stride size for height
    :param activation_fn: Active function, default set as ReLu
    :return: Output of the layer, weights and biases
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name+'weights', initializer=data[name+'_w'], trainable=training)
        biases = tf.get_variable(name+"bias", initializer=data[name+'_b'], trainable=training)
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        tf.summary.histogram(name + '/weight', weights)
        tf.summary.histogram(name + '/biases', biases)
        _activation_summary(conv, name)
        return activation, weights, biases


def fully_connected_f(input_tensor, name, data, training, activation_fn=tf.nn.relu):
    """
    Definition of the fully connected layer for finetune
    :param input_tensor: Input layer with the shape of [batchSiz, width, height, channel]
    :param name: Name of this layer
    :param data: Data used for initialization
    :param training: Flag for training
    :param activation_fn: Active function, default set as ReLu
    :return: Output of the layer, weights and biases
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', initializer=data[name+'_w'], trainable=training)
        biases = tf.get_variable("bias", initializer=data[name+'_b'], trainable=training)
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        tf.summary.histogram(name + '/weight', weights)
        tf.summary.histogram(name + '/biases', biases)
        logits = activation_fn(logits)
        _activation_summary(logits, name)
        return logits, weights, biases


def pool(input_tensor, name, kh, kw, dh, dw):
    """
    Definition of the pooling layer
    :param input_tensor: Input layer with the shape of [batchSiz, width, height, channel]
    :param name: Name of this layer
    :param kw: Width of the convolutional layer
    :param kh: Height of the convolutional layer
    :param dw: Stride size for width
    :param dh: Stride size for height
    :return: Output of the layer
    """
    return tf.nn.max_pool(input_tensor, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='VALID', name=name)


def loss(logits, onehot_labels):
    """
    Definition of the loss function
    :param logits: Input layer with the shape of [batchSiz, width, height, channel]
    :param onehot_labels: One-Hot encoding
    :return: Output of the layer
    """
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return loss