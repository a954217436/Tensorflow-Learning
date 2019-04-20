# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:15:48 2019

@author: Sean
"""

import tensorflow as tf


#创建一个常量op
m1 = tf.constant([[3, 3]])
#创建一个常量op
m2 = tf.constant([[2], [3]])
#创建一个矩阵乘法op，把m1和m2传入
product = tf.matmul(m1, m2)

print(product)

##定义一个会话，启动默认图
#sess = tf.Session()
##调用sess的run方法来执行矩阵乘法op
##run(product)触发了图中三个op
#result = sess.run(product)
#print(result)
#sess.close()

#等同于以上操作，自动关闭
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
