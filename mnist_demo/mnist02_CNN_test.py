#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-14 12:15:25
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import tensorflow as tf
from PIL import Image,ImageFilter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 手写图像预处理，转换为模型输入格式(像素为0-1，黑底白字，灰度)
def imagePrepare():
    file_path = './test_images/7.png'
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

# 初始化w、b
def  weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial)

# 卷积和池化
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1, 2, 2,1],padding="SAME")

#待识别图像的像素值
input_pixels = imagePrepare()
# None表示可以是任何维度
x = tf.placeholder(tf.float32,[None,784])

x_image = tf.reshape(x,[-1,28,28,1])

############ 第一层CNN ###############
# 28*28  --->   24*24  --->   12*12
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# input
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)

############ 第二层CNN ###############
# 12*12   --->   8*8   --->  7*7

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

############ 全连接层 ###############
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

############ Dropout 减少过拟合 ###############
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

############ 输出层 ###############
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

result = tf.argmax(y_conv,1)

saver = tf.train.Saver()
checkpoint_dir = './model/mnist02_CNN/'
with tf.Session() as sess:
    # sess.run(init)
    # 使用之前的参数
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass
    result_num = sess.run(result,feed_dict={x:[input_pixels], keep_prob: 1.0}) #x加[]是为了满足模型中x的维度。
    print('预测结果为: '+str(result_num))
