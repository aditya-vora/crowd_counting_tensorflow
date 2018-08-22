"""
Standard layer definations.

1) conv: Defines a convolutional layer and initializes the weights and biases.
2) pool: Defines a pooling layer which reduces the dimension of the input to half.
3) loss: Defines a loss layer to compute the mean square pixel wise error.

@author: Aditya Vora
"""

import tensorflow as tf
import numpy as np

def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
    """
    Convolution layer
    :param input_tensor: Input Tensor (feature map / image)
    :param name: name of the convolutional layer
    :param kw: width of the kernel
    :param kh: height of the kernel
    :param n_out: number of output feature maps
    :param dw: stride across width
    :param dh: stride across height
    :param activation_fn: nonlinear activation function
    :return: output feature map after activation
    """
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=(kh, kw, n_in, n_out), stddev=0.01), dtype=tf.float32, name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out]), dtype=tf.float32, name='biases')
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        tf.summary.histogram("weights", weights)
        return activation

def pool(input_tensor, name, kh, kw, dh, dw):
    """
    Max Pooling layer
    :param input_tensor: input tensor (feature map) to the pooling layer
    :param name: name of the layer
    :param kh: height scale down size. (Generally 2)
    :param kw: width scale down size. (Generally 2)
    :param dh: stride across height
    :param dw: stride across width
    :return: output tensor (feature map) with reduced feature size (Scaled down by 2).
    """
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss(est, gt):
    """
    Computes mean square error between the network estimated density map and the ground truth density map.
    :param est: Estimated density map
    :param gt: Ground truth density map
    :return: scalar loss after doing pixel wise mean square error.
    """
    return tf.losses.mean_squared_error(est, gt)

# Module to test the loss layer
if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 20, 20, 1])
    y = tf.placeholder(tf.float32, [1, 20, 20, 1])
    mse = loss(x, y)
    sess = tf.Session()
    dict = {
        x: 5*np.ones(shape=(1,20,20,1)),
        y: 4*np.ones(shape=(1,20,20,1))
    }
    print sess.run(mse, feed_dict=dict)
