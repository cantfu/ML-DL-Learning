#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-12 14:42:47
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
# @Desc    : 用CNN做的模型  初始化权重 ReLU
#
import tensorflow as tf
from numpy import float32
import datetime
from tensorflow.examples.tutorials.mnist import input_data

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

start = datetime.datetime.now()
# 载入数据
mnist = input_data.read_data_sets('./data',one_hot=True)


# None表示可以是任何维度
x = tf.placeholder(tf.float32,[None,784])
# y:正确值(正确label)
y = tf.placeholder(tf.float32, [None, 10])

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


# 损失函数 loss function
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))

# ADAM优化器做梯度下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 找出预测正确的标签 tf.equal()返回的是true or false 数组
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
# 求正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#存储训练的模型
saver = tf.train.Saver(max_to_keep=1) 

# 一批次大小
batch_size = 50
# 所有批次数量
n_batch = mnist.train.num_examples // batch_size
with tf.Session() as sess:
    saver_max_acc = 0
    sess.run(tf.initialize_all_variables())
    for epoch in range(1):
        for i in range(n_batch):
            batch = mnist.train.next_batch(50)
            sess.run(train_step,feed_dict={x:batch[0], y: batch[1], keep_prob: 0.5})
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
                print("step %d acc : %g"%(i,train_accuracy))
        
                acc=accuracy.eval(feed_dict={x: mnist.test.images[0:2000], y: mnist.test.labels[0:2000], keep_prob: 1.0})
                print("test %d accuracy %g"%(epoch,acc))
                # 添加判断语句，选择保存精度最高的模型
                if acc > saver_max_acc:
                    saver.save(sess,'./model/mnist02_CNN/mnist02_CNN.ckpt',global_step=epoch+1)
                    saver_max_acc = acc

end = datetime.datetime.now()
print((end - start).seconds)