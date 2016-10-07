#-*- coding: utf-8 -*-

# Lab1-2, 將正確率調製趨近 0.99

# Improve The Evaluate Accuracy

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib


FLAGS = None


"""
Initializing variables to zero, when the activation of
a layer is made of ReLUs will yield a null gradient, this generates
dead precisely a ReLU is not differentiable in 0, but it
is differentiable in any epsilon bubble defined around 0.
"""

# init for a weight variable
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# init for a bias variable
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
input: ...
"""
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
"""
value: ...
"""
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Values that we will input during the computation
  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])
  
  # reshape vectors of size 784, to squares of size 28x28
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  """1st layer: convolutional layer with max pooling"""
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  """2st layer: convolutional layer with max pooling"""
  # [width, height, depth, output_size]
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  """3st layer: fully connect layer"""
  # [input_size, output_size]
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  # flattening the output of the previous layer
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  """Add fropout"""
  #using a placeholder for keep_prob will allow to turn off the dropout during testing
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  """4th layer: fully connected layer"""
  # [input_size, output_size]
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # y = tf.matmul(x, W) + b
  # W = tf.Variable(tf.zeros([784, 10]))
  # b = tf.Variable(tf.zeros([10]))

  # # Define loss and optimizer
  # y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y))
  
  # # SGD for minimizing the cross-entropy (learning rate = 0.5)
  # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  # Start the new session & Initialize the variables
  sess = tf.InteractiveSession()
  tf.initialize_all_variables().run()

  # Define the accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  # Train
  for n in range(20000):
    batch = mnist.train.next_batch(50)
    if n % 100 == 0:
      train_accuracy= accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
      print ("step %d, training accuracy %g" % (n, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

  # # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  print("test accuracy % g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
  # # A red/black/blue colormap
  # cdict = {'red':[(0.0,  1.0,  1.0),
  #                 (0.25, 1.0,  1.0),
  #                 (0.5,  0.0,  0.0),
  #                 (1.0,  0.0,  0.0)],
  #        'green':[(0.0,  0.0,  0.0),
  #                 (1.0,  0.0,  0.0)],
  #         'blue':[(0.0,  0.0,  0.0),
  #                 (0.5,  0.0,  0.0),
  #                 (0.75, 1.0,  1.0),
  #                 (1.0,  1.0,  1.0)]}
  # redblue = matplotlib.colors.LinearSegmentedColormap('red_black_blue',cdict,256)
  
  # wts = W.eval(sess)
  # for i in range(0,10):
  #   im = wts.flatten()[i::10].reshape((28,-1))
  #   plt.imshow(im, cmap = redblue, clim=(-1.0, 1.0))
  #   plt.colorbar()
  #   print("Digit %d" % i)
  #   plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
