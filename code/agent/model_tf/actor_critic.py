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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp
import warnings


class ActorCriticNetwork(keras.Model):
    def __init__(self, num_action=8, base_model_dir='../model/base_model', load_ac_model=False,
                 ac_model_dir='../model/ac_model', norm_rate=0.25, optimizer=None, ):
        super(ActorCriticNetwork, self).__init__()
        self.num_action = num_action
        self.norm_rate = norm_rate
        self.optimizer = optimizer
        # init model_dir
        self.base_model_dir = base_model_dir
        self.base_ckpt_dir = os.path.join(self.base_model_dir, 'ckpt')
        self.ac_model_dir = ac_model_dir
        self.ac_ckpt_dir = os.path.join(self.ac_model_dir, 'ckpt')
        if load_ac_model:
            self.ac_model = self.__load_ac_model__()
        else:
            self.ac_model = self.__init_ac_model__(self.norm_rate)
        self.ac_model = self.__compile_ac_model__()
        # self.target_ac_model = self.update_target_ac_model()
    
    def __init_ac_model__(self, norm_rate):
        print('-' * 20, 'Initializing a new actor-critic model!', '-' * 20, )
        base_model = tf.keras.models.load_model(self.base_ckpt_dir)
        print('-' * 20, base_model, '-' * 20, )
        base_model.summary()
        
        feature_extraction = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        # sub_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        
        for i, layer in enumerate(feature_extraction.layers):
            layer.trainable = False
        
        if base_model.output_shape[-1] != self.num_action:
            warnings.warn('The output_shape of base_model is different from num_action!\
                 Randomly initialized parameters will be used!')
            dense = Dense(self.num_action, name='action_dense', kernel_constraint=max_norm(norm_rate)
                          )(feature_extraction.output)
            action_pred = Activation('softmax', name='softmax')(dense)
        else:
            action_pred = base_model.output
        value_pred = Dense(1, name='value_dense', kernel_constraint=max_norm(norm_rate))(feature_extraction.output)
        ac_model = Model(inputs=base_model.input, outputs=[value_pred, action_pred, ])
        print('-' * 20, ac_model, '-' * 20, )
        ac_model.summary()
        
        return ac_model
    
    def __load_ac_model__(self):
        print('-' * 20, 'Loading pre-trained actor-critic model!', '-' * 20, )
        return tf.keras.models.load_model.load(self.ac_ckpt_dir)
    
    def update_target_ac_model(self):
        self.target_ac_model = deepcopy(self.ac_model)
        # keras.models.clone_model(model)
        # model.get_weights()
        # model.set_weights(weights)
        return self.target_ac_model
    
    def save_model(self):
        print('-' * 20, 'Saving actor-critic model!', '-' * 20, )
        self.ac_model.save(self.ac_ckpt_dir)
    
    def __compile_ac_model__(self):
        self.ac_model.compile(optimizer=self.optimizer, )
        return self.ac_model
    
    def call(self, state, training=False, **kwargs):
        return self.ac_model(state, training=training, **kwargs)


### ??????step??????next_obs_batch, reward_batch, done_batch, info_batch =  env.step(actions_batch)

if __name__ == '__main__':
    
    ac_model = ActorCriticNetwork()
    print('Hello World!')
