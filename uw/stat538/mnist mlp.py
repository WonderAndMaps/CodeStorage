# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:44:36 2018

@author: dell
"""

"""
MLP
"""


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()



x = tf.placeholder("float", shape=[None, 784]) 
y_ = tf.placeholder("float", shape=[None, 10]) #10 = 0~9 one hot

                   
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



#first layer
W_1 = weight_variable([28*28, 1600])
b_1 = bias_variable([1600])
x_vec = tf.reshape(x, [-1,28*28])
L_1 = tf.nn.relu(tf.add(tf.matmul(x_vec, W_1), b_1))


#second layer
W_2 = weight_variable([1600, 1600])
b_2 = bias_variable([1600])
L_2 = tf.nn.relu(tf.add(tf.matmul(L_1, W_2), b_2))


#softmax
V = weight_variable([1600, 10])
y_pred=tf.nn.softmax(tf.matmul(L_2, V))


#cross_entropy = -tf.reduce_sum(y_*tf.log(y_pred))
log_loss = tf.losses.log_loss(y_,y_pred)
train_step = tf.train.AdamOptimizer(1e-4).minimize(log_loss)
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
train_accuracy = 0;i=0

while train_accuracy<0.975:
  i += 1
  batch = mnist.test.next_batch(20)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:mnist.test.images, y_: mnist.test.labels})
    print ("step",i,"\t training accuracy",train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels})

testLayer0=mnist.test.images.reshape([10000,-1])
testLayer1=sess.run(L_1, feed_dict={x:mnist.test.images}).reshape([10000,-1])
testLayer2=sess.run(L_2, feed_dict={x:mnist.test.images}).reshape([10000,-1])
testLayer3=sess.run(y_pred, feed_dict={x:mnist.test.images}).reshape([10000,-1])

import pandas as pd
pd.DataFrame(testLayer0).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlp0.txt')
pd.DataFrame(testLayer1).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpL1.txt')
pd.DataFrame(testLayer2).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpL2.txt')
pd.DataFrame(testLayer3).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpL3.txt')

pd.DataFrame(mnist.test.labels).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpLabel.txt')

