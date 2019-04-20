# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:16:28 2019

@author: Sean
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 使用numpy生成200个随机点
# 生成200个均匀分布的点，再增加一个维度
x_data = np.linspace(-5, 5, 200)[:, np.newaxis]
noise = np.random.normal(0,0.1,x_data.shape)
#y_data = np.cos(x_data) + noise
y_data = np.sin(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#神经元个数
Neurons = 10

#定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, Neurons]))
bias_L1 = tf.Variable(tf.zeros([1, Neurons]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + bias_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)    #双曲正切函数非线性激活函数

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([Neurons, 1]))
bias_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + bias_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)    #双曲正切函数非线性激活函数

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for neuron in enumerate([10, 20, 30, 40, 50]):
        for i in range(2000):
            sess.run(train_step, feed_dict={x:x_data, y:y_data})
    
        #获得预测值
        prediction_value = sess.run(prediction, feed_dict={x:x_data})
        mse = np.mean(np.square(y_data - prediction_value))
        print('Neurons: %s, MSE: %s'%(neuron[1], mse))
        #画图
        plt.figure()
    #    plt.plot(x_data, y_data, 'r-')
    #    plt.plot(x_data, prediction_value, 'g-')
        plt.scatter(x_data, y_data, s=10)
        plt.plot(x_data, prediction_value, 'r-', lw=4)
        plt.title('Neurons: %d'%neuron[1])
        plt.show()
        Neurons = neuron


















