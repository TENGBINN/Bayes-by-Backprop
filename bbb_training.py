# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:33:55 2017

@author: TENG
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

import bbb

def main(argv=None):
    
    mnist_data = mnist.input_data.read_data_sets("./datasets/MNIST_data", one_hot=True)
    
    conf = bbb.Config()
    conf.layer_nodes = [200, 100]
    conf.batch_size = 128
    conf.example_size = mnist_data.train.num_examples
    conf.sample_times = 3
    conf.multi_mu = [0.0, 0.0]
    conf.multi_sigma = [np.exp(-1.0, dtype=np.float32), np.exp(-6.0, dtype=np.float32)]
    conf.multi_ratio = [0.25, 0.75]
    conf.target_onehot = True
    conf.learning_rate_base = 0.001
    conf.learning_rate_decay = 0.99
    # conf.optimizer_api = "GradientDescentOptimizer"
    conf.optimizer_api = "AdamOptimizer"
    
    data = tf.placeholder(tf.float32, [None, 28*28])
    target = tf.placeholder(tf.float32, [None, 10])
    nn = bbb.BBB(data, target, conf, is_training=True)
    
    model_name = "model"
    model_save_path = "./model"
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(300000):
            xs, ys = mnist_data.train.next_batch(conf.batch_size)

            _, loss_value, step, lr = sess.run(
                [nn.optimize, nn.loss, nn.global_step, nn.learning_rate], 
                feed_dict={nn.data: xs, nn.target: ys})
            
            if i % 1000 == 0:
                print("After %d training step(s), lr(%.6f), loss on training batch is %g." % (step, lr, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=nn.global_step)
                
    # writer = tf.summary.FileWriter("./log/modified_mnist_train.log", tf.get_default_graph())
    # writer.close()

if __name__ == "__main__":
    tf.app.run()
    
    