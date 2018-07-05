# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:49:34 2018

@author: fuyang
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()


#通过为输入图像和目标输出类别创建节点，来开始构建计算图
#784 = 28*28 None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定。
x = tf.placeholder("float", shape=[None, 784]) 
y_ = tf.placeholder("float", shape=[None, 10]) #10 = 0~9 one hot

                   
#权重在初始化时加入少量的噪声来打破对称性以及避免0梯度。
#使用的是ReLU神经元，用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）                   
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
#池化用简单传统的2x2大小的模板做max pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#第一层卷积
#卷积在每个5x5的patch中算出100个特征。卷积的权重张量形状是[5, 5, 1, 100]，
#前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 
#而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 100])
b_conv1 = bias_variable([100])

#把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
#(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积层
W_conv2 = weight_variable([5, 5, 100, 100])
b_conv2 = bias_variable([100])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#全连接层 1024个神经元的全连接层
#图片大小为7*7=28*28经过两次2*2得来maxpool
#（卷积时padding=‘SAME’，在输入图外延加n圈0使得卷积后大小不变）
W_fc1 = weight_variable([7 * 7 * 100, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*100])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Dropout
#防止过拟合，placeholder表示神经元保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#输出 softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#训练模型、准确率
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
log_loss = tf.losses.log_loss(y_,y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(log_loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
train_accuracy = 0;i=0

while train_accuracy<0.975:
  i += 1
  batch = mnist.test.next_batch(20)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print ("step",i,"\t training accuracy",train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

accuracy.eval(feed_dict={
        x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

"""
print ("test accuracy",accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
#提取结果
result = sess.run(tf.argmax(y_conv,1),feed_dict={x: mnist.test.images, keep_prob:1.0})
trueLabel = sess.run(tf.argmax(y_,1),feed_dict={y_: mnist.test.labels})
wrongLabel = [i for i,j in enumerate(result!=trueLabel) if j==True] #错判的图片号

#看看
import matplotlib.pyplot as plt  
def display_digit(num,result):
    label = list(mnist.test.labels[num,:]).index(1.0)
    image = mnist.test.images[num,:].reshape([28,28])
    plt.title(['Example:',num, 'Label:',label, 'Predict:', result[num]])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
display_digit(wrongLabel[25],result)
"""

#三层的东西
testLayer0=mnist.test.images.reshape([10000,-1])
testLayer1=sess.run(h_pool1, feed_dict={x:mnist.test.images}).reshape([10000,-1])
testLayer2=sess.run(h_pool2, feed_dict={x:mnist.test.images}).reshape([10000,-1])
testLayer3=sess.run(y_conv, feed_dict={x:mnist.test.images, keep_prob:1.0}).reshape([10000,-1])


import pandas as pd
pd.DataFrame(testLayer0).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL0.txt')
pd.DataFrame(testLayer1).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL1.txt')
pd.DataFrame(testLayer2).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL2.txt')
pd.DataFrame(testLayer3).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL3.txt')
pd.DataFrame(mnist.test.labels).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnLabel.txt')



