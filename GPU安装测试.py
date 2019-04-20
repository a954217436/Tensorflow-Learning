# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:52:13 2019

@author: Sean
"""
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

