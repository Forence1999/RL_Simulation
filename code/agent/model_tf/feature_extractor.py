# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation_D3QN
# @File: feature_extractor.py
# @Time: 2022/02/02/21:18
# @Software: PyCharm
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda, Add, BatchNormalization, Reshape
import tensorflow_probability as tfp


def ResCNN_feature_extractor(model_dir='../../model/base_model/'):
    '''
    initialize a feature extractor for a d3qn RL agent.
    '''
    ckpt_dir = os.path.join(model_dir, 'feature_extractor','ckpt')
    fe = tf.keras.models.load_model(ckpt_dir)
    fe.trainable = False
    fe.compile()
    
    print('-' * 20, 'ResCNN_feature_extractor', '-' * 20, )
    fe.summary()
    
    return fe


if __name__ == '__main__':
    print('Hello World!')
    
    fe = ResCNN_feature_extractor()
    
    print('Brand-new World!')
