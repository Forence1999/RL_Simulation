# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: A2C
# @File: A2C.py
# @Time: 2021/10/29/22:54
# @Software: PyCharm
import os
import sys
import time
import random
import numpy as np
from copy import deepcopy
import collections
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda, Add
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp
import warnings


class D3QNNetwork(keras.Model):
    def __init__(self, num_action=8, base_model_dir='../model/base_model', load_d3qn_model=False,
                 d3qn_model_dir='../model/d3qn_model', norm_rate=0.25, dueling=True, ):
        super(D3QNNetwork, self).__init__()
        self.num_action = num_action
        self.norm_rate = norm_rate
        self.dueling = dueling
        # init model_dir
        self.base_model_dir = base_model_dir
        self.base_ckpt_dir = os.path.join(self.base_model_dir, 'ckpt')
        self.d3qn_model_dir = d3qn_model_dir
        self.d3qn_ckpt_dir = os.path.join(self.d3qn_model_dir, 'ckpt')
        if load_d3qn_model:
            self.d3qn_model = self.__load_d3qn_model__()
        else:
            self.d3qn_model = self.__init_d3qn_model__(self.norm_rate)
    
    def __init_d3qn_model__(self, norm_rate):
        print('-' * 20, 'Initializing a new D3QN model!', '-' * 20, )
        base_model = tf.keras.models.load_model(self.base_ckpt_dir)
        print('-' * 20, base_model, '-' * 20, )
        base_model.summary()
        
        feature_extraction = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        # sub_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        
        for i, layer in enumerate(feature_extraction.layers):
            layer.trainable = False
        
        if self.dueling:
            state_value = Dense(1, )(feature_extraction.output)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.num_action,))(state_value)
            
            action_advantage = Dense(self.num_action, )(feature_extraction.output)
            # kernel_constraint=max_norm(norm_rate))(feature_extraction.output)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                      output_shape=(self.num_action,))(action_advantage)
            output = Add()([state_value, action_advantage])
        else:
            output = Dense(self.num_action, activation="linear", )(feature_extraction.output)
        
        d3qn_model = Model(inputs=base_model.input, outputs=output)
        
        print('-' * 20, 'd3qn_model', '-' * 20, )
        d3qn_model.summary()
        
        return d3qn_model
    
    def __load_d3qn_model__(self):
        try:
            print('-' * 20, 'Loading pre-trained D3QN model!', '-' * 20, )
            return tf.keras.models.load_model(self.ac_ckpt_dir)
        except:
            print('-' * 20, 'Fail to load the pre-trained D3QN model! Initialized model will be used for D3QN!',
                  '-' * 20, )
            return self.__init_d3qn_model__()
    
    def save_model(self, model_path=None):
        model_path = self.d3qn_ckpt_dir if (model_path is None) else os.path.join(model_path, 'ckpt')
        print('-' * 20, 'Saving D3QN model to %s!' % model_path, '-' * 20, )
        self.d3qn_model.save(model_path)
    
    def get_weights(self):
        return self.d3qn_model.get_weights()
    
    def set_weights(self, weights):
        return self.d3qn_model.set_weights(weights)
    
    def compile(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.d3qn_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    
    def predict(self, x, **kwargs):
        return self.d3qn_model.predict(x, **kwargs)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, **kwargs):
        self.d3qn_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)
    
    def call(self, state, training=False, **kwargs):
        return self.d3qn_model(state, training=training, **kwargs)


### 编写step接口next_obs_batch, reward_batch, done_batch, info_batch =  env.step(actions_batch)

if __name__ == '__main__':
    
    d3qn_model = D3QNNetwork()
    print('Hello World!')
