# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: agent.py
# @Time: 2021/10/31/10:19
# @Software: PyCharm

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda, Add, BatchNormalization, Reshape


def split_model(model_dir='../model/base_model_fullData_woBN'):
    full_ckpt = os.path.join(model_dir, 'full_model', 'ckpt')
    fe_ckpt = os.path.join(model_dir, 'feature_extractor', 'ckpt')
    cls_ckpt = os.path.join(model_dir, 'classifier', 'ckpt')
    
    full_model = tf.keras.models.load_model(full_ckpt)
    print('-' * 20, 'full_model', '-' * 20, )
    full_model.summary()
    
    fe = Model(inputs=full_model.input, outputs=full_model.get_layer('feature_linear').output)
    print('-' * 20, 'feature_extractor', '-' * 20, )
    fe.summary()
    
    cls = full_model.get_layer('output_conv')
    print('-' * 20, 'classifier', '-' * 20, )
    cls.summary()
    
    tf.keras.models.save_model(model=fe, filepath=fe_ckpt)
    tf.keras.models.save_model(model=cls, filepath=cls_ckpt)


if __name__ == '__main__':
    print('Hello World!')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    split_model()
    
    print('Brand-new World!')
