# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: A2C
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


class D3QNNetwork(keras.Model):
    def __init__(self, num_action=8, dueling=True, base_model_dir='../../model/base_model',
                 load_d3qn_model=False, d3qn_model_dir='../../model/d3qn_model', ):
        super(D3QNNetwork, self).__init__()
        self.num_action = num_action
        self.dueling = dueling
        # init model_dir
        self.base_model_dir = base_model_dir
        self.base_ckpt_dir = os.path.join(self.base_model_dir, 'ckpt')
        self.d3qn_model_dir = d3qn_model_dir
        self.d3qn_ckpt_dir = os.path.join(self.d3qn_model_dir, 'ckpt')
        if load_d3qn_model:
            self.d3qn_model = self.__load_d3qn_model__()
        else:
            self.d3qn_model = self.__init_d3qn_model__()
    
    def __init_d3qn_model__(self, ):
        print('-' * 20, 'Initializing a new D3QN model...', '-' * 20, )
        base_model = tf.keras.models.load_model(self.base_ckpt_dir)
        print('-' * 20, base_model, '-' * 20, )
        # base_model.summary()
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('feature_linear').output)
        # sub_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        
        for layer in feature_extractor.layers:
            layer.trainable = False
        
        if self.dueling:
            '------------------------------------- value network -------------------------------------------------'
            state_value = Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', use_bias=True,
                                 name='value_conv')(feature_extractor.output)
            # state_value = BatchNormalization(axis=-1)(state_value)
            state_value = Reshape((-1,))(state_value)
            state_value = Dense(1, name='value_dense')(state_value)
            # state_value = Lambda(lambda s: tf.expand_dims(s[:, 0], -1), output_shape=(self.num_action,))(state_value)
            
            '------------------------------------- advantage network -------------------------------------------------'
            # adv = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='adv_conv_0')(
            #     feature_extraction.output)
            # adv = Activation('relu')(adv)
            # adv = Conv2D(1, kernel_size=(30, 1), strides=(1, 1), padding='valid', name='adv_conv_1')(adv)
            # action_advantage = Reshape((-1,))(adv)
            
            action_advantage = base_model.get_layer('output_linear').output
            
            # kernel_constraint=max_norm(norm_rate))(feature_extraction.output)
            # action_advantage = Lambda(lambda a: a[:, :] - tf.mean(a[:, :], keepdims=True),
            #                           output_shape=(self.num_action,))(action_advantage)
            '------------------------------------- add -------------------------------------------------'
            output = Add()([state_value, action_advantage])
        else:
            output = base_model.get_layer('output_linear').output
        
        d3qn_model = Model(inputs=base_model.input, outputs=output)
        
        print('-' * 20, 'd3qn_model', '-' * 20, )
        d3qn_model.summary()
        
        return d3qn_model
    
    def __load_d3qn_model__(self, ):
        try:
            print('-' * 20, 'Loading pre-trained D3QN model...', '-' * 20, )
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # from tensorflow.keras.utils import plot_model
    # model = ResCNN_4_STFT_DOA(num_classes=8, num_time_clips=30, num_res_block=2, num_filter=64)
    # model.build(input_shape=(None, 30, 508, 8))
    # model.summary()
    # model.predict(np.random.random((1, 30, 508, 8)))
    # lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.001,  # alpha=1e-7,
    #                                                 decay_steps=220 * 50)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc', ])
    #
    # weight_path = '../../model/0/ckpt'
    # model.load_weights(filepath=weight_path)
    #
    # tf.keras.models.save_model(model, weight_path, )
    d3qn_model = D3QNNetwork()
    print('Hello World!')
