# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:38:11 2019

@author: Sean
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:07:31 2019

@author: Sean
"""

#测试集正确率0.97
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#每个批次大小
batch_size = 128
#计算一共多少个批次
n_batch = mnist.train.num_examples // batch_size


#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(0.01 * tf.truncated_normal([784, 10], mean=0.0, stddev=0.1), name='W')
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([10])+0.1, name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b) 

with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.square(y - prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #保存到log目录下，若不存在则新建该文件夹
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        #每100个图片训练一次
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter: " + str(epoch) + ", Testing Accuracy: " + str(acc))

























