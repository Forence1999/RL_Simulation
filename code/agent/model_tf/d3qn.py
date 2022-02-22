# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: D3QN
# @File: A2C.py
# @Time: 2021/10/29/22:54
# @Software: PyCharm
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda, Add, BatchNormalization, Reshape
import tensorflow_probability as tfp


class D3QN_Classifier(keras.Model):
    '''
    initialize a classifier for a d3qn RL agent.
    '''
    
    def __init__(self, num_action=8, dueling=True, base_model_dir='../../model/base_model/',
                 load_d3qn_model=False, d3qn_model_dir='../../model/d3qn_model/', ):
        super(D3QN_Classifier, self).__init__()
        
        self.num_action = num_action
        self.dueling = dueling
        
        self.base_model_dir = base_model_dir
        self.base_ckpt_dir = os.path.join(self.base_model_dir, 'classifier', 'ckpt')
        self.d3qn_model_dir = d3qn_model_dir
        self.d3qn_ckpt_dir = os.path.join(self.d3qn_model_dir, 'classifier', 'ckpt')
        if load_d3qn_model:
            self.d3qn_model = self.__load_d3qn_model__()
        else:
            self.d3qn_model = self.__init_d3qn_model__()
    
    def __init_d3qn_model__(self, ):
        print('-' * 20, 'Initializing a new D3QN model...', '-' * 20, )
        base_classifier = tf.keras.models.load_model(self.base_ckpt_dir)
        
        # input = Input(name='feature_input')  # Input(shape=(num_time_clips, stft_len, 8), name='input')
        
        if self.dueling:
            '------------------------------------- value network -------------------------------------------------'
            state_value = Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True,
                                 name='value_conv')(base_classifier.input)
            # state_value = BatchNormalization(axis=-1)(state_value)
            state_value = Reshape((-1,), name='value_reshape')(state_value)
            state_value = Dense(1, name='value_dense')(state_value)
            
            '------------------------------------- advantage network -------------------------------------------------'
            action_advantage = base_classifier.output
            # kernel_constraint=max_norm(norm_rate))(feature_extraction.output)
            
            '------------------------------------- add -------------------------------------------------'
            output = Add()([state_value, action_advantage])
        else:
            output = base_classifier.output
        
        d3qn_model = Model(inputs=base_classifier.input, outputs=output)
        
        print('-' * 20, 'd3qn_classifier', '-' * 20, )
        d3qn_model.summary()
        
        return d3qn_model
    
    def __load_d3qn_model__(self, ):
        try:
            print('-' * 20, 'Loading pre-trained D3QN classifier...', '-' * 20, )
            return tf.keras.models.load_model(self.d3qn_ckpt_dir)
        except Exception as e:
            print('Warning:', e)
            print('-' * 20, 'Fail to load the pre-trained D3QN model! An initialized model will be used for D3QN!',
                  '-' * 20, )
            return self.__init_d3qn_model__()
    
    def save_model(self, model_path=None, ):
        model_path = self.d3qn_ckpt_dir if (model_path is None) else os.path.join(model_path, 'ckpt')
        print('-' * 20, 'Saving D3QN model to %s...' % model_path, '-' * 20, )
        self.d3qn_model.save(model_path)
    
    def get_weights(self, ):
        return self.d3qn_model.get_weights()
    
    def set_weights(self, weights, ):
        return self.d3qn_model.set_weights(weights)
    
    def compile(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.d3qn_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    
    def predict(self, x, **kwargs):
        return self.d3qn_model.predict(x, **kwargs)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, **kwargs):
        self.d3qn_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)
    
    def call(self, state, training=False, **kwargs):
        return self.d3qn_model(state, training=training, **kwargs)


if __name__ == '__main__':
    print('Hello World!')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    d3qn_model = D3QN_Classifier()
    
    print('Brand-new World!')
