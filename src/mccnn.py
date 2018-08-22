"""
Network defination for multi-column Convolutional Neural Network.

Network contains 3 columns with different receptive fields in order to model crowd at different perspectives.
Contains one fuse layer which concatenates different column outputs and fuses the features with a learning 1x1 filters.

For more info on the architecture please refer this paper:
[1] Zhang, Yingying, et al. "Single-image crowd counting via multi-column convolutional neural network."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

@author: Aditya Vora
"""
import tensorflow as tf
import layers as L
import numpy as np

def shallow_net_9x9(x):
    net = L.conv(x, name="conv_sn9x9_1", kh=9, kw=9, n_out=16)
    net = L.pool(net, name="pool_sn9x9_1", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn9x9_2", kw=7, kh=7, n_out=32)
    net = L.pool(net, name="pool_sn9x9_2", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn9x9_3", kw=7, kh=7, n_out=16)
    net = L.conv(net, name="conv_sn9x9_4", kw=7, kh=7, n_out=8)
    return net

def shallow_net_7x7(x):
    net = L.conv(x, name="conv_sn7x7_1", kh=7, kw=7, n_out=20)
    net = L.pool(net, name="pool_sn7x7_1", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn7x7_2", kw=5, kh=5, n_out=40)
    net = L.pool(net, name="pool_sn7x7_2", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn7x7_3", kw=5, kh=5, n_out=20)
    net = L.conv(net, name="conv_sn7x7_4", kw=5, kh=5, n_out=10)
    return net

def shallow_net_5x5(x):
    net = L.conv(x, name="conv_sn5x5_1", kh=5, kw=5, n_out=24)
    net = L.pool(net, name="pool_sn5x5_1", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn5x5_2", kw=3, kh=3, n_out=48)
    net = L.pool(net, name="pool_sn5x5_2", kh=2, kw=2, dw=2, dh=2)
    net = L.conv(net, name="conv_sn5x5_3", kw=3, kh=3, n_out=24)
    net = L.conv(net, name="conv_sn5x5_4", kw=3, kh=3, n_out=12)
    return net

def fuse_layer(x1, x2, x3):
    x_concat = tf.concat([x1, x2, x3],axis=3)
    return L.conv(x_concat, name="fuse_1x1_conv", kw=1, kh=1, n_out=1)


def build(input_tensor, norm = False):
    """
    Builds the entire multi column cnn with 3 shallow nets with different input kernels and one fusing layer.
    :param input_tensor: Input tensor image to the network.
    :return: estimated density map tensor.
    """
    tf.summary.image('input', input_tensor, 1)
    if norm:
        input_tensor = tf.cast(input_tensor, tf.float32) * (1. / 255) - 0.5
    net_1_output = shallow_net_9x9(input_tensor)                # For column 1 with large receptive fields
    net_2_output = shallow_net_7x7(input_tensor)                # For column 2 with medium receptive fields
    net_3_output = shallow_net_5x5(input_tensor)                # For column 3 with small receptive fields
    full_net = fuse_layer(net_1_output, net_2_output, net_3_output) # Fusing all the column output features
    return full_net


# Testing the data flow of the network with some random inputs.
if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 200, 300, 1])
    net = build(x)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    d_map = sess.run(net,feed_dict={x:255*np.ones(shape=(1,200,300,1), dtype=np.float32)})
    prediction = np.asarray(d_map)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=2)
