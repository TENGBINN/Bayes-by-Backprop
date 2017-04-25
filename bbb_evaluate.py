# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:59:45 2017

@author: TENG
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

import bbb

def main(argv=None):
    
    mnist_data = mnist.input_data.read_data_sets("./datasets/MNIST_data", one_hot=True)
    
    conf = bbb.Config()
    conf.layer_nodes = [200, 100]
    conf.example_size = mnist_data.validation.num_examples
    conf.target_onehot = True
    # conf.optimizer_api = "GradientDescentOptimizer"
    conf.optimizer_api = "AdamOptimizer"
    
    data = tf.placeholder(tf.float32, [None, 28*28])
    target = tf.placeholder(tf.float32, [None, 10])
    validate_feed = {
        data: mnist_data.validation.images, 
        target: mnist_data.validation.labels}
        
    nn = bbb.BBB(data, target, conf, is_training=False)
    
    model_save_path = "./model"

    saver = tf.train.Saver()
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(nn.accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
            
        time.sleep(10)
    

if __name__ == "__main__":
    tf.app.run()
    
    