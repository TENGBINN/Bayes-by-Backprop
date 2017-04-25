# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:44:09 2017

@author: TENG
"""

import functools
import numpy as np
import tensorflow as tf


class Config:
    """ BBB Settings """

    layer_nodes = [500, 300]
    batch_size = 20
    example_size = 55000
    sample_times = 3
    multi_mu = [0.0, 0.0]
    multi_sigma = [np.exp(-1.0, dtype=np.float32), np.exp(-6.0, dtype=np.float32)]
    multi_ratio = [0.25, 0.75]
    target_onehot = True
    target_class_size = 10
    learning_rate_base = 0.001
    learning_rate_decay = 0.99
    optimizer_api = "AdamOptimizer"
    
    
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class BBB:
    """ Bayes By Backprop """

    def __init__(self, data, target, config, is_training):
        
        self._conf = config
        self._data = data
        self._target = target
        self._target_onehot = target
        self._is_training = is_training

        # self._input_size
        # self._output_size
        
        self.config
        self.data
        self.target

        self._weights = []
        self._biases = []
        self.params

        if self._is_training:
            self.global_step
            self.learning_rate
            self.loss
            self.optimize
        else:
            self.inference
            self.accuracy
            
    @lazy_property
    def config(self):
        self._conf.optimizer_api = tf.train.__dict__[self._conf.optimizer_api]
        return self._conf
            
    @lazy_property
    def data(self):
        data = self._data
        self._input_size = int(self._data.get_shape()[-1])
        return data
        
    @lazy_property
    def target(self):
        target = self._target
        if self._conf.target_onehot:
            self._output_size = int(self._target.get_shape()[-1])
            self._target = tf.argmax(self._target, -1)
        else:
            self._output_size = self._conf.target_class_size
            self._target_onehot = tf.reshape(
                tf.one_hot(self._target, depth=self._conf.target_class_size), 
                [-1, self._conf.target_class_size])
        return target
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    def _get_layer_sizes(self):
        input_size = self._input_size
        output_size = self._output_size
        layer_sizes = list(zip([input_size] + self._conf.layer_nodes, 
                              self._conf.layer_nodes + [output_size]))
        return layer_sizes
    
    def _get_params(self, prefix, shape):
        if prefix == "weights":
            n = shape[0]
            mus = tf.get_variable(
                prefix+"_mus", shape, dtype=tf.float32, 
                initializer=tf.truncated_normal_initializer(stddev=1/np.sqrt(n)))
            rhos = tf.get_variable(
                prefix+"_rhos", shape, dtype=tf.float32, 
                initializer=tf.constant_initializer(np.log(np.exp(1/np.sqrt(n)) - 1)))
            return mus, rhos
        elif prefix == "biases":
            biases = tf.get_variable(
                prefix, shape, dtype=tf.float32, 
                initializer=tf.constant_initializer(0.01))
            return biases
    
    def _get_weights(self, mus, rhos=None):
        if rhos is None:
            return mus
        else:
            epsilons = tf.random_normal(shape=mus.get_shape())
            weights = mus + tf.log(1 + tf.exp(rhos)) * epsilons
            return weights
    
    @lazy_property
    def params(self):
        layer_sizes = self._get_layer_sizes()
        for idx, shape in enumerate(layer_sizes):
            with tf.variable_scope("layer%d" % (idx+1)):
                weights_mus, weights_rhos = self._get_params("weights", shape)
                self._weights.append({"mus": weights_mus, "rhos": weights_rhos})
                biases = self._get_params("biases", shape[-1])
                self._biases.append(biases)
                
    def _layer(self, idx, inputs, is_training=False, with_active=False):
        weights = self._get_weights(
            self._weights[idx]["mus"], 
            self._weights[idx]["rhos"] if is_training else None)
        biases = self._biases[idx]
        outputs = tf.matmul(inputs, weights) + biases
        if with_active:
            outputs = tf.nn.relu(outputs)
        if is_training:
            self._weights[idx]["weights"] = weights
        return outputs
        
    def _inference(self, is_training):
        layers = [self._data]
        layer_sizes = self._get_layer_sizes()
        for i in range(len(layer_sizes)):
            layer = self._layer(i, layers[-1], is_training=is_training, 
                                with_active=i<len(layer_sizes)-1)
            layers.append(layer)
        return layers[-1]
    
    def _gaussian(self, xs, mus, sigmas):
        return tf.exp(- tf.square(xs - mus) / (2 * tf.square(sigmas))) / (tf.sqrt(2*np.pi) * tf.abs(sigmas))
        
    def _log_p(self, xs, multi_mu, multi_sigma, multi_ratio):
        p = tf.constant(0.0)
        for i in range(len(multi_ratio)):
            p += multi_ratio[i] * tf.clip_by_value(self._gaussian(xs, multi_mu[i], multi_sigma[i]), 1e-10, 1.0)
        logps = tf.log(p)
        return tf.reduce_sum(logps)
    
    def _log_q(self, xs, mus, rhos):
        sigmas = tf.log(1 + tf.exp(rhos))
        logqs = tf.log(tf.clip_by_value(self._gaussian(xs, mus, sigmas), 1e-10, 1.0))
        return tf.reduce_sum(logqs)

    def _loss_ER(self, y):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y, labels=self._target_onehot))
        return cross_entropy
    
    def _loss_KL(self):
        loss = tf.constant(0.0)
        layer_sizes = self._get_layer_sizes()
        for i in range(len(layer_sizes)):
            loss += self._log_q(self._weights[i]["weights"], 
                                self._weights[i]["mus"], 
                                self._weights[i]["rhos"])
            loss -= self._log_p(self._weights[i]["weights"], 
                                self._conf.multi_mu, 
                                self._conf.multi_sigma, 
                                self._conf.multi_ratio)
            loss -= self._log_p(self._biases[i], 
                                self._conf.multi_mu, 
                                self._conf.multi_sigma, 
                                self._conf.multi_ratio)
        return loss / self._conf.example_size
    
    @lazy_property
    def loss(self):
        loss = tf.constant(0.0)
        for s in range(self._conf.sample_times):
            y = self._inference(is_training=True)
            loss += self._loss_ER(y) + self._loss_KL()
        return loss / self._conf.sample_times
    
    @lazy_property
    def global_step(self):
        with tf.variable_scope("global_step"):
            global_step = tf.Variable(0, trainable=False)
        return global_step
    
    @lazy_property
    def learning_rate(self):
        self._learning_rate = tf.train.exponential_decay(
            self._conf.learning_rate_base,
            self.global_step,
            self._conf.example_size / self._conf.batch_size, 
            self._conf.learning_rate_decay,
            staircase=True)
        return self._learning_rate
    
    @lazy_property
    def optimize(self):
        train_step = self._conf.optimizer_api(self._learning_rate).minimize(
            self.loss, global_step=self.global_step, name='optimize')
        return train_step
    
    @lazy_property
    def inference(self):
        return self._inference(is_training=False)

    @lazy_property
    def accuracy(self):
        y = self.inference
        correct_prediction = tf.equal(tf.argmax(y, 1), self._target)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    