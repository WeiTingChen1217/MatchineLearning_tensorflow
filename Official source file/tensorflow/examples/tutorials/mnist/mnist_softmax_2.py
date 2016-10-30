#-*- coding: utf-8 -*-

# Lab1, 正確跑完 MNIST for ML Beginners, accuracy target: 0.90 up

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


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # tmp = mnist.train.images[2]
  # tmp = tmp.reshape((28,28))
  # 
  # plt.imshow(tmp, cmap = cm.Greys)
  # plt.show()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  
  # SGD for minimizing the cross-entropy (learning rate = 0.5)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Initialize the variables
  sess = tf.InteractiveSession()
  
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  # A red/black/blue colormap
  cdict = {'red':[(0.0,  1.0,  1.0),
                  (0.25, 1.0,  1.0),
                  (0.5,  0.0,  0.0),
                  (1.0,  0.0,  0.0)],
         'green':[(0.0,  0.0,  0.0),
                  (1.0,  0.0,  0.0)],
          'blue':[(0.0,  0.0,  0.0),
                  (0.5,  0.0,  0.0),
                  (0.75, 1.0,  1.0),
                  (1.0,  1.0,  1.0)]}
  redblue = matplotlib.colors.LinearSegmentedColormap('red_black_blue',cdict,256)
  
  wts = W.eval(sess)
  for i in range(0,10):
    im = wts.flatten()[i::10].reshape((28,-1))
    plt.imshow(im, cmap = redblue, clim=(-1.0, 1.0))
    plt.colorbar()
    print("Digit %d" % i)
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
