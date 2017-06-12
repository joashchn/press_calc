# -- coding: utf-8 --
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
import press_data

# Parameters
# learning_rate = 0.001
learning_rate = 0.0001
# initial_learning_rate = 0.001
training_iters = 400000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 512  # MNIST data input (img shape: 28*28)
n_classes = 100  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
y = tf.placeholder(tf.float32, [None, 100])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    # return tf.nn.sigmoid(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 512, 1, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=4)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=4)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # fc1 = tf.concat([fc1, y], 1)

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)
    # # fc2 = tf.nn.sigmoid(fc2)
    # # Apply Dropout
    # fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([8, 1, 1, 16]), name='press_wc1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([8, 1, 16, 32]), name='press_wc2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32 * 32, 1024]), name='press_wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='press_wout')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16]), name='press_bc1'),
    'bc2': tf.Variable(tf.random_normal([32]), name='press_bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='press_bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='press_bout')
}

def test(wav):
    #sess=tf.InteractiveSession()
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    wav_list = []
    wav_arr = []
    for i in range(len(wav.split(","))):
        wav_arr.append(float(wav.split(",")[i]))
    wav_list.append(wav_arr)

    wav_in = numpy.array(wav_list)
    wav_in = (wav_in-wav_in.min())/(wav_in.max()-wav_in.min())
    # print(wav_in)

    pred = conv_net(x, weights, biases, keep_prob)

    # Initializing the variables
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(
        [weights['wc1'], weights['wc2'], weights['wd1'], weights['out'], biases['bc1'], biases['bc2'], biases['bd1'],
         biases['out']])
    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        sess.run(init)

        # saver.restore(sess, "data/press/press_cnn.ckpt")
        saver.restore(sess, "data/press/press_cnn_re.ckpt")
        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        prediction=sess.run(pred, feed_dict={x: wav_in[:1], keep_prob: 1.})
        # print(tf.argmax(prediction, 1))
        # return str('This is a with possibility %.6f' % (prediction[:, 0]))
        id = int(sess.run(tf.argmax(prediction, 1))[0])
        result = str("[%d]<br>" % id)
        for i in range(0, n_classes):
            if i == id:
                result += str('%d: <b><font color=red>%.6f</font></b><br>' % (i, prediction[:, i]))
            else:
                result += str('%d: %.6f<br>' % (i, prediction[:, i]))
        return result

def train(mnist):
    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step=training_iters,
    #                                            decay_steps=1000000, decay_rate=0.1)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # cost_rmse = tf.sqrt(tf.reduce_mean(tf.cast(tf.square(pred - y), tf.float32)))
    # cost = tf.reduce_mean(tf.cast(tf.square(pred - y), tf.float32))
    # cost = tf.reduce_sum(tf.squared_difference(y, pred))
    # cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred - y), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # 使用梯度下降法，设置步长0.1，来最小化损失

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # correct_pred = (tf.square(pred - y) < 0.015)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(
        [weights['wc1'], weights['wc2'], weights['wd1'], weights['out'], biases['bc1'], biases['bc2'], biases['bd1'],
         biases['out']])

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                #print(batch_x);
            step += 1
        print("Optimization Finished!")
        saver.save(sess, "data/press/press_cnn_re.ckpt")

        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:500],
                                          y: mnist.test.labels[:500],
                                          keep_prob: 1.}))
# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = press_data.read_data_sets("/tmp/data/", one_hot=True)
    train(mnist)

# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数。
if __name__ == '__main__':
    tf.app.run()
