# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: agent.py
# @Time: 2021/10/31/10:19
# @Software: PyCharm

import os
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from model_tf.PER import Memory
from model_tf.d3qn import D3QNNetwork
from collections import deque
import pylab


class DQNAgent(object):
    def __init__(self, num_action=8, batch_size=8, memory_size=64, ddqn=True, softUpdate=True, lr=0.001,
                 dueling=True, eps_decay=True, usePER=False, d3qn_model_dir='../model/d3qn_model',
                 load_d3qn_model=True, d3qn_model_name=None, base_model_dir='../model/base_model', ):
        assert usePER == False, 'PER has not been checked'
        # space
        self.num_action = num_action
        
        # defining model parameters
        self.ddqn = ddqn  # use double deep q network
        self.softUpdate = softUpdate  # use soft parameter update
        self.dueling = dueling  # use dealing netowrk
        self.eps_decay = eps_decay  # use epsilon greedy strategy
        self.usePER = usePER  # use priority experienced replay
        self.TAU = 0.1  # target network soft update hyperparameter
        self.gamma = 0.95  # discount rate
        self.lr = lr
        
        # exploration hyperparameters for epsilon and epsilon greedy strategy
        self.eps = 1.0  # exploration probability at start
        self.eps_min = 0.01  # minimum exploration probability
        self.eps_decay_rate = 0.999  # exponential decay rate for exploration prob
        self.name = self.__gen_name__() if (d3qn_model_name is None) else d3qn_model_name
        
        # Instantiate memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        if self.usePER:
            self.MEMORY = Memory(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        self.load_d3qn_model = load_d3qn_model
        self.base_model_dir = base_model_dir
        self.d3qn_model_dir = os.path.join(d3qn_model_dir, self.name)
        os.makedirs(self.d3qn_model_dir, exist_ok=True)
        
        # create main model and target model
        self.model = D3QNNetwork(num_action=self.num_action, load_d3qn_model=self.load_d3qn_model, dueling=self.dueling,
                                 base_model_dir=self.base_model_dir, d3qn_model_dir=self.d3qn_model_dir, )
        if self.ddqn:
            self.target_model = D3QNNetwork(num_action=self.num_action, load_d3qn_model=self.load_d3qn_model,
                                            dueling=self.dueling,
                                            base_model_dir=self.base_model_dir, d3qn_model_dir=self.d3qn_model_dir, )
    
    def __gen_name__(self, ):
        '''
        generate a RL model model name for saving.
        :return:
        '''
        dueling = 'Dueling' if self.dueling else ''
        dqn = 'DDQN' if self.ddqn else 'DQN'
        softUpdate = 'softUpdate' if self.softUpdate else ''
        eps_decay = 'epsDecay' if self.eps_decay else ''
        usePER = 'usePER' if self.usePER else ''
        lr = 'lr_' + str(self.lr)  # TODO
        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
        
        name = '_'.join((dueling, dqn, softUpdate, eps_decay, usePER, lr, time_stamp)).replace('__', '_')
        print('-' * 20, 'Model Name:', name, '-' * 20, )
        
        return name
    
    def save_model(self, model_path=None, ):
        '''
        save the RL model. If ddqn, save target_model, else save model.
        :param model_path:
        :return:
        '''
        model_path = self.d3qn_model_dir if (model_path is None) else model_path
        if self.ddqn:
            self.target_model.save_model(model_path)
        else:
            self.model.save_model(model_path)
    
    def compile(self, **kwargs):
        '''
        compile the RL model.
        :param kwargs:
        :return:
        '''
        # lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=1e4)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=decay_steps, decay_rate=0.99, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss="mean_squared_error", metrics=["accuracy"], **kwargs)
    
    def update_target_model(self, ):
        '''
        bases on softUpdate, update target_model
        :return:
        '''
        if self.ddqn and (not self.softUpdate):
            self.target_model.set_weights(self.model.get_weights())
        elif self.ddqn and self.softUpdate:
            model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            for idx, (weight, target_weight) in enumerate(zip(model_theta, target_model_theta)):
                target_weight = target_weight * (1 - self.TAU) + weight * self.TAU
                target_model_theta[idx] = target_weight
            self.target_model.set_weights(target_model_theta)
    
    def remember(self, state, action, reward, state_, done, ):
        '''
        save the experience to memory buffer.
        '''
        experience = state, action, reward, state_, done
        if self.usePER:
            self.MEMORY.append(experience)
        else:
            self.memory.append((experience))
    
    def remember_batch(self, batch_experience, useDiscount=True):
        '''
        save a batch of experience to memory buffer.
        if discount: apply discount to the reward.
        '''
        if useDiscount:
            for i in range(len(batch_experience) - 2, -1, -1):
                batch_experience[i][2] += self.gamma * batch_experience[i + 1][2]
        for experience in batch_experience:
            if self.usePER:
                self.MEMORY.append(experience)
            else:
                self.memory.append((experience))
    
    def replay(self, ):
        '''
        experience replay and learn on a batch
        '''
        if self.usePER:
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # minibatch = np.array(minibatch, dtype=object)
        # state, action, reward, next_state, done = minibatch.T
        # state, next_state = np.concatenate(state), np.concatenate(next_state)
        # action = np.asarray(action, dtype=np.int32)
        # reward = np.asarray(reward, dtype=np.float64)
        # done = np.asarray(done, dtype=np.bool)
        state, action, reward, state_, done = list(zip(*minibatch))
        # for i_state, i_action, i_reward, i_state_, i_done in range(len(minibatch)):
        #     state.append(i_state), action.append(i_action), reward.append(i_reward), done.append(i_done)
        #     state_.append(i_state_ if (i_state_ is not None) else i_state)
        target_old, target = self.learn_per_batch(state, action, reward, state_, done)
        
        if self.usePER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, action] - target[indices, action])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
    
    def learn_per_batch(self, state, action, reward, state_, done, ):
        '''
        optimizer the model for one batch
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param done:
        :return:
        '''
        state, state_ = np.asarray(state), np.asarray(state_)
        action = np.asarray(action, dtype=np.int32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.bool)
        
        target = self.model.predict(state)  # predict Q for starting state with the main network
        target_old = np.array(target)
        target_next = self.model.predict(state_)  # predict best action in ending state with the main network
        
        for i in range(len(done)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    target_value = self.target_model.predict(state_)
                    # predict Q for ending state with the target network
                    a = np.argmax(target_next[i])  # current Q Network selects the action
                    target[i][action[i]] = reward[i] + self.gamma * (target_value[i][a])  # target Q Network to evaluate
                else:  # Standard - DQN ---- DQN chooses the max Q value among next actions
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        
        self.model.fit(state, target, batch_size=self.batch_size, verbose=2)  # Train the Neural Network with batches
        
        return target_old, target
    
    def learn(self, state, action, reward, state_, done, ):
        self.learn_per_batch([state], [action], [reward], [state_], [done], )
    
    def act(self, state, decay_step, ):
        '''
        return the action and explore_prob based on eps_decay
        :param state:
        :param decay_step:
        :return:
        '''
        # EPSILON GREEDY STRATEGY
        if self.eps_decay:  # Improved version of epsilon greedy strategy for Q-learning
            explore_prob = self.eps_min + (self.eps - self.eps_min) * self.eps_decay_rate ** decay_step
        else:
            explore_prob = self.eps
        
        if explore_prob > random.random():  # Make a random action (exploration)
            return random.randrange(self.num_action), explore_prob
        else:  # Get action from Q-network (exploitation)
            return np.argmax(self.predict(state)), explore_prob
    
    def predict(self, state, ):
        '''
        produce the output of the given state with model.
        '''
        state = np.array([state])
        return self.model.predict(state)


if __name__ == '__main__':
    
    agent = DQNAgent()
    print('Hello World!')
