'''
@Date  : 2017-12-13 10:56
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    使用tf-slim构建 cnn网络， 用于识别mnist.
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

BATCH_SIZE = 50
STEPS = 20000
LEARNING_RATE = 1e-4

MNIST_DATA_PATH = '../../MNIST_data'


def get_next_batch(mnist, get_train_batch=True):
    if get_train_batch:
        batch = mnist.train.next_batch(BATCH_SIZE)
    else:
        batch = mnist.test.next_batch(BATCH_SIZE)

    images, labels = batch[0], batch[1]
    images = images.reshape([-1,28,28,1])

    return images, labels


# 使用 conv2d时，会自动加上bias
def inference(net,is_train=True):
    if is_train:
        # keep_prob 改为 0.5 会比 0.8好非常多。
        keep_prob = 0.5
    else:
        keep_prob = 1
    with tf.variable_scope('train', 'train', [net], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0004), kernel_size=[5, 5], stride=1):
            with slim.arg_scope([slim.max_pool2d], kernel_size=[2,2], stride=2):
                net = slim.conv2d(net, 32, scope='conv1')
                net = slim.max_pool2d(net, scope='pool1')

                net = slim.conv2d(net, 64, scope='conv2')
                net = slim.max_pool2d(net, scope='pool2')

                net = slim.flatten(net)
                net = slim.dropout(net, keep_prob=keep_prob, scope='dropout')
                net = slim.fully_connected(net, 10)
                probs = slim.softmax(net)
                return probs


def get_loss(probs,labels):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(probs),
                                                  reduction_indices=[1]))
    return cross_entropy


if __name__ == '__main__':
    mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    with tf.Session() as sess:
        images_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_placeholder')
        labels_palceholder = tf.placeholder(tf.float32, [None, 10], name='labels_placeholder')
        # 用于训练得到的预测概率，probs的计算中加入了 dropout
        probs = inference(images_placeholder)
        # 用于预测的概率，无dropout
        probs_predict = inference(images_placeholder, is_train=False)

        loss = get_loss(probs, labels_palceholder)

        # 将自定义的损失loss加入总损失
        tf.losses.add_loss(loss)
        # 得到所有损失值，包括正则损失
        total_loss = tf.losses.get_total_loss()
        optimaizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_step = slim.learning.create_train_op(total_loss, optimaizer)

        correct_prediction = tf.equal(tf.argmax(probs_predict,1), tf.argmax(labels_palceholder,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #这里注意，由于train中定义了一些变量，一次init应当位于train的后面。
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            inputs, labels = get_next_batch(mnist)
            sess.run(train_step, feed_dict={images_placeholder: inputs, labels_palceholder: labels})
            if i%10==0:
                test_input, test_labels = get_next_batch(mnist, False)
                print('step: %d, corss entropy: %.3f' % (i, sess.run(loss, feed_dict={images_placeholder: inputs, labels_palceholder: labels})))
                print('step: %d, predict accuracy: %.3f' %
                      (i, sess.run(accuracy, feed_dict={images_placeholder: test_input, labels_palceholder: test_labels})))

