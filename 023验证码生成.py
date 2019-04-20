# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:34:03 2019

@author: Sean
"""

from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys

number = [str(i) for i in range(10)]
alphabet = [chr(i) for i in range(97, 123)]
alphabet_CAP = [chr(i) for i in range(65, 91)]

def random_captcha_text(char_set=number, captcha_size=4):
    #验证码列表
    captcha_text = []
    for i in range(captcha_size):
        #随机选择
        c = random.choice(char_set)
        #加入验证码列表
        captcha_text.append(c)
    return captcha_text


#生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=160, height=60)
    #获得随机生成的验证码
    captcha_text = random_captcha_text(number)
#    captcha_text = random_captcha_text(alphabet + alphabet_CAP)
    #转换为字符串
    captcha_text = ''.join(captcha_text)
    #写到文件
    image.write(captcha_text, 'captcha/tests/' + captcha_text + '.jpg')

gen_captcha_text_and_image()

#数量小于10000，因为重名了
num = 50
if __name__ == '__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>Creating image %d/%d'%(i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('生成完毕')












