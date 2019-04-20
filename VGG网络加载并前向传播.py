# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:20:14 2019

@author: Sean
"""

import tensorflow as tf
import scipy.io
import scipy.misc
import os
import matplotlib.pyplot as plt
import numpy as np


def _conv_layer(_input, weights, bias):
    conv = tf.nn.conv2d(_input, weights, strides=(1,1,1,1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(_input):
    return tf.nn.max_pool(_input, ksize=(1,2,2,1), strides=(1,2,2,1),
                          padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def net(data_path, input_image):
    layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
            )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # mat文件中与TensorFlow中kernel代表的顺序不一样
            kernels = np.transpose(kernels, (1,0,2,3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'pool':
            current = _pool_layer(current)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers


VGG_PATH = 'D:/image_retraining/imagenet-vgg-verydeep-19.mat'
IMG_PATH = 'D:/image_retraining/testimage/pet3.jpg'

input_image = imread(IMG_PATH)
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])

with tf.Session() as sess:
    image = tf.placeholder(tf.float32, shape=shape)
    nets, mean_pixel, all_layers = net(VGG_PATH, image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    layers = all_layers
    for i, layer in enumerate(layers):
        print("[%d/%d] %s" % (i+1, len(layers), layer))
        features = nets[layer].eval(feed_dict={image: input_image_pre})
        
        print("Type of 'features' is ", type(features))
        print("Shape of 'features' is %s" % (features.shape,))
        if i == 35:
#            plt.figure(i+1, figsize = (10, 5))
#            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i+1)
#            plt.title("" + layer)
#            plt.colorbar()
#            plt.show()
#            plt.imsave('D:/image_retraining/VGG_features/'+'%d'%(i+1)+layer+'.jpg', features[0, :, :, 0])
            for j in range(512):
                plt.imsave('D:/image_retraining/VGG_features_final512/'+'%d'%(j+1)+'.jpg', features[0, :, :, j])
                

















