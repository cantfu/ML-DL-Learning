#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-12 11:15:08
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
# RGB转灰度----> L = R * 0.299 + G * 0.587+ B * 0.114
# 发现粗笔画的能识别，细笔画的识别很差
import tensorflow as tf
from PIL import Image,ImageFilter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 手写图像预处理，转换为模型输入格式(像素为0-1，黑底白字，灰度)
def imagePrepare():
    file_path = './test_images/3.png'
    # 读取图片,转为灰度
    im = Image.open(file_path).convert('L')
    # im = origin_im.resize(28,28)
    pixels1 = list(im.getdata())

    print('pixels1:'+str(len(pixels1)))
    # 转为黑底白字(灰度图中0表示黑色)
    pixels2 = [(255-x)*1.0/255.0 for x in pixels1]
    print('pixels2:'+ str(len(pixels2)))

    img = np.array(pixels2).reshape((28,28))
    plt.imshow(img,cmap='gray',interpolation="nearest")
    plt.axis("off")        #关闭坐标轴
    plt.show()

    return pixels2
#待识别图像的像素值
input_pixels = imagePrepare()
# None表示可以是任何维度
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 线性模型 x*W+b
prediction = tf.nn.softmax(tf.matmul(x,W)+b)
result = tf.argmax(prediction,1)
# 初始化变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    # sess.run(init)
    # 使用之前的参数
    saver.restore(sess,'./model/mnist01/mnist01.ckpt')
    print('b:' , sess.run(b))
    result_num = sess.run(result,feed_dict={x:[input_pixels]}) #x加[]是为了满足模型中x的维度。
    print('预测结果为: '+str(result_num))