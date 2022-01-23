# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: agent.py
# @Time: 2021/10/31/10:19
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
from model_tf.actor_critic import ActorCriticNetwork
from model_tf.PER import Memory
from model_tf.d3qn import D3QNNetwork
from collections import deque
import pylab


class DQNAgent:
    def __init__(self, num_action=8, batch_size=8, memory_size=64, ddqn=True, Soft_Update=True, lr=0.001,
    
                 dueling=True, epsilon_greedy=True, USE_PER=True, d3qn_model_dir='../model/d3qn_model',
                 load_d3qn_model=True):
        # space
        # self.state_size = state_size
        self.num_action = num_action
        
        # defining model parameters
        self.ddqn = ddqn  # use doudle deep q network
        self.Soft_Update = Soft_Update  # use soft parameter update
        self.dueling = dueling  # use dealing netowrk
        self.epsilon_greedy = epsilon_greedy  # use epsilon greedy strategy
        self.USE_PER = USE_PER  # use priority experienced replay
        self.TAU = 0.1  # target network soft update hyperparameter
        self.gamma = 0.95  # discount rate
        self.lr = lr
        
        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob
        self.name = self.__gen_name__()
        # Instantiate memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        if self.USE_PER:
            self.MEMORY = Memory(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        self.load_d3qn_model = load_d3qn_model
        self.d3qn_model_dir = os.path.join(d3qn_model_dir, self.name)
        os.makedirs(self.d3qn_model_dir, exist_ok=True)
        
        # create main model and target model
        self.model = D3QNNetwork(num_action=self.num_action, load_d3qn_model=self.load_d3qn_model,
                                 d3qn_model_dir=self.d3qn_model_dir, dueling=self.dueling, )
        if self.ddqn:
            self.target_model = D3QNNetwork(num_action=self.num_action, load_d3qn_model=self.load_d3qn_model,
                                            d3qn_model_dir=self.d3qn_model_dir, dueling=self.dueling, )
    
    def __gen_name__(self):
        dueling = 'Dueling_' if self.dueling else ''
        dqn = 'DDQN' if self.ddqn else 'DQN'
        softupdate = '_SoftUpdate' if self.Soft_Update else ''
        greedy = '_Greedy' if self.epsilon_greedy else ''
        PER = '_PER' if self.USE_PER else ''
        lr = '_lr_' + str(0.0001)  # TODO
        name = dueling + dqn + softupdate + greedy + PER + lr
        print('-' * 20, 'Model Name:', name, '-' * 20, )
        
        return name
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.d3qn_model_dir
        if self.ddqn:
            self.target_model.save_model(model_path)
        else:
            self.model.save_model(model_path)
    
    def compile(self, **kwargs):
        # lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=1e4)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=decay_steps, decay_rate=0.99, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss="mean_squared_error", metrics=["accuracy"], **kwargs)
    
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        elif self.Soft_Update and self.ddqn:
            model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            for idx, (weight, target_weight) in enumerate(zip(model_theta, target_model_theta)):
                target_weight = target_weight * (1 - self.TAU) + weight * self.TAU
                target_model_theta[idx] = target_weight
            self.target_model.set_weights(target_model_theta)
    
    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.append(experience)
        else:
            self.memory.append((experience))
    
    def replay(self):
        if self.USE_PER:
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # minibatch = np.array(minibatch, dtype=object)
        # state, action, reward, next_state, done = minibatch.T
        # state, next_state = np.concatenate(state), np.concatenate(next_state)
        # action = np.asarray(action, dtype=np.int32)
        # reward = np.asarray(reward, dtype=np.float64)
        # done = np.asarray(done, dtype=np.bool)
        state, action, reward, next_state, done = [], [], [], [], []
        for i in range(len(minibatch)):
            state.append(minibatch[i][0])
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state.append(minibatch[i][3] if minibatch[i][3][0] is not None else minibatch[i][0])
            done.append(minibatch[i][4])
        state, next_state = np.asarray(state), np.asarray(next_state)
        action = np.asarray(action, dtype=np.int32)
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=np.bool)
        
        target_old, target = self.learn_per_batch(state, action, reward, next_state, done)
        
        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, action] - target[indices, action])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
    
    def learn_per_batch(self, state, action, reward, state_, done):
        state, state_ = np.asarray(state), np.asarray(state_)
        action = np.asarray(action, dtype=np.int32)
        reward = np.asarray(reward)
        done = np.asarray(done, dtype=np.bool)
        
        target = self.model.predict(state)  # predict Q-values for starting state using the main network
        target_old = np.array(target)
        target_next = self.model.predict(state_)  # predict best action in ending state using the main network
        if self.ddqn:
            target_val = self.target_model.predict(state_)  # predict Q-values for ending state using the target network
        
        for i in range(len(done)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    a = np.argmax(target_next[i])  # current Q Network selects the action
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])  # target Q Network to evaluate
                else:  # Standard - DQN ---- DQN chooses the max Q value among next actions
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)  # Train the Neural Network with batches
        
        return target_old, target
    
    def learn(self, state, action, reward, state_, done):
        self.learn_per_batch([state], [action], [reward], [state_], [done])
    
    def act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        if self.epsilon_greedy:  # Improved version of epsilon greedy strategy for Q-learning
            explore_prob = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        else:  # OLD EPSILON STRATEGY
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_prob = self.epsilon
        
        if explore_prob > np.random.rand():  # Make a random action (exploration)
            return random.randrange(self.num_action), explore_prob
        else:  # Get action from Q-network (exploitation)
            state = np.array(state)[np.newaxis, :]
            return np.argmax(self.model.predict(state)), explore_prob
    
    def predict(self, state):
        np.argmax(self.model.predict(state))


if __name__ == '__main__':
    
    agent = DQNAgent()
    print('Hello World!')
    
    
    def PlotModel(self, score, episode):
        pylab.figure(figsize=(18, 9))
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        if self.dueling:
            dueling = '_Dueling'
        if self.epsilon_greedy:
            greedy = '_Greedy'
        if self.USE_PER:
            PER = '_PER'
        try:
            pylab.savefig(dqn + self.env_name + softupdate + dueling + greedy + PER + "_CNN.png")
        except OSError:
            pass
        
        return str(self.average[-1])[:5]
    
    
    def run(self):
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            i = 0
            while not done:
                decay_step += 1
                action, explore_prob = self.act(state, decay_step)
                next_state, reward, done, _ = self.step(action)
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every REM_STEP update target model
                    if e % self.REM_STEP == 0:
                        self.update_target_model()
                    
                    # every episode, plot the result
                    average = self.PlotModel(i, e)
                    
                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i,
                                                                                    explore_prob, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model to", self.Model_name)
                        # self.save(self.Model_name)
                        break
                self.replay()
        self.env.close()
    
    
    def test(self):
        self.load(self.Model_name)
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break
