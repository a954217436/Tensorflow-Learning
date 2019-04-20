# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:02:51 2019

@author: Sean
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#输入图片是28*28
n_inputs = 28    #一次输入一行，即28个数
max_time = 28    #一共输入28行
lstm_size = 100    #隐层单元
n_classes = 10    #10个分类
batch_size = 50    #每批次50个样本
n_batch = mnist.train.num_examples // batch_size


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


#定义RNN网络
def RNN(X, weights, biases):
    
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    #定义LSTM基本cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state 
    # final_state[1]是hidden state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

#计算RNN返回结果
prediction = RNN(x, weights, biases)

#损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter: "+str(epoch)+",Test Acc = "+str(acc))


















