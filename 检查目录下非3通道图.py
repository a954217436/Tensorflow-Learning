# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:43:28 2019

@author: Sean
"""

#encoding:utf-8
from PIL import Image
import os
 
def get_not_rgb_images(rootdir):
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        filename = os.path.join(rootdir, list[i])
        # print(filename)
        
        if os.path.isfile(filename):
            img = Image.open(filename)
            pixels = img.getpixel((0, 0))
 
            if type(pixels) == int:
                print('单通道:' + filename)
#                os.remove(filename)
            elif type(pixels) == tuple:
                if  len(pixels) != 3:
                    print('非RGB的多通道:' +filename)
        else:
            get_not_rgb_images(filename)
 
 
if __name__ == '__main__':
    rootdir = r'D:\image_retraining\slim\images'
    get_not_rgb_images(rootdir)
