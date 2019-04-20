# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:46:53 2019

@author: Sean
"""

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import cv2


lines = tf.gfile.GFile('D:/image_retraining/retrain/output_labels.txt').readlines()
uid_to_human = {}
for uid, line in enumerate(lines):
#    line = line[:-1] 去掉换行符，同下
    line = line.strip('\n')
    uid_to_human[uid] = line


def id_to_string(uid):
    if uid not in uid_to_human:
        return ''
    return uid_to_human[uid]


#创建一个图来存放google训练好的模型
with tf.gfile.GFile(r'D:\image_retraining\retrain\output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name = '')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    #遍历目录
    for (root, dirs, files) in os.walk(r'D:\image_retraining\testimage'):
        for file in files:
            image_data = tf.gfile.GFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            
            #图片分类结果按置信度排序得到序号top：
            top = predictions.argsort()[::-1]
            print(top, predictions[0])
            human_string = id_to_string(top[0])
            print('Pred: ' + human_string)
            print('File: ' + file)
            print('^' * 30)
            
            img = cv2.imread(os.path.join(root, file))
            cv2.putText(img, human_string, (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255))
            cv2.imshow(file, img)
            cv2.imwrite(root+'/test_pred/'+file, img)
            

cv2.waitKey(0)
cv2.destroyAllWindows()












