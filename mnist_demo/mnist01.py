#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-09 19:00:06
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
# @Desc    : 线性回归模型-->Testing Accuracy 0.9096

import tensorflow as tf
from numpy import float32
import datetime
from tensorflow.examples.tutorials.mnist import input_data

start = datetime.datetime.now()
# 载入数据
mnist = input_data.read_data_sets('./data',one_hot=True)

# 一批次大小
batch_size = 200
# 所有批次数量
n_batch = mnist.train.num_examples // batch_size

# None表示可以是任何维度
x = tf.placeholder(tf.float32,[None,784])
# y:正确值(正确label)
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 线性模型 x*W+b
prediction = tf.nn.softmax(tf.matmul(x,W)+b)


# 损失函数 loss function
# loss = tf.reduce_mean(tf.square(y - prediction))
cross_entropy = -tf.reduce_sum(prediction * tf.log(y))


# 使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)

# 找出预测正确的标签 tf.equal()返回的是true or false 数组
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
# 求正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,float32))


#存储训练的模型
saver = tf.train.Saver(max_to_keep=1) 

# 循环训练30次
with tf.Session() as sess:
    sess.run(init)
    saver_max_acc = 0
    for epoch in range(500):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            # xs = mnist.train.images
            # ys = mnist.train.labels
            # sess.run(train_step,feed_dict={x:xs,y:ys})


        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images, y:mnist.test.labels})
        print("Iter:"+str(epoch)+",Testing Accuracy "+str(acc))
        # 添加判断语句，选择保存精度最高的模型
        if acc > saver_max_acc:
            saver.save(sess,'./model/mnist01/mnist01.ckpt',global_step=epoch+1)
            saver_max_acc = acc

end = datetime.datetime.now()
print((end - start).seconds)

# if __name__ == '__main__':
#     main()
