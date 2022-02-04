# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation_D3QN
# @File: temp.py
# @Time: 2022/02/01/19:58
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
from collections import deque

import tensorflow as tf
import tensorflow_probability as tfp

from agent_models import FeatureExtractor, SAC_actor, SAC_critic

EPS = np.finfo(float).eps


class SACAgent(object):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """
    
    def __init__(self, num_Q=2, num_action=8, reward_scale=1.0, reward_discount_rate=0.99,
                 policy_lr=3e-4, Q_lr=3e-4, alpha_lr=3e-4,
                 usePER=False, batch_size=8, memory_size=64, learnTimes=1,
                 softUpdate=True, softUpdate_tau=5e-3, target_update_interval=1,
                 base_model_dir='../model/base_model', sac_model_dir='../model/d3qn_model',
                 load_sac_model=True, sac_model_name=None, based_on_base_model=True,
                 **kwargs):
        super(SACAgent, self).__init__(**kwargs)
        assert softUpdate == True, 'softUpdate must be True for SAC'
        assert usePER == False, 'PER has not been created'
        if sac_model_name is None:
            self.name = self.__class__.__name__
            print('Warning:',
                  'A SACAgent class is initialized with default name (\'SACAgent\'). And it may load unexpected models.')
        else:
            self.name = sac_model_name
        
        # space
        self.num_action = num_action
        # reward
        self.reward_scale = reward_scale
        self.discount_rate = reward_discount_rate
        # optimize
        self.learnTimes = learnTimes
        self.softUpdate = softUpdate  # use soft parameter update
        self.tau = softUpdate_tau  # target network soft update hyperparameter
        self.policy_lr = policy_lr
        self.Q_lr = Q_lr
        self.alpha_lr = alpha_lr
        self.target_update_interval = target_update_interval
        self.target_entropy = self.__heuristic_target_entropy__(self.num_action)
        # memory
        self.usePER = usePER
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        # model
        self.num_Q = num_Q
        self.load_sac_model = load_sac_model
        self.base_model_dir = base_model_dir
        self.based_on_base_model = based_on_base_model
        self.sac_model_dir = os.path.join(sac_model_dir, self.name)
        os.makedirs(self.sac_model_dir, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor(model_dir=self.base_model_dir, )
        self.policy = SAC_actor(base_model_dir=base_model_dir,
                                load_sac_model=load_sac_model, sac_model_dir=sac_model_dir,
                                based_on_base_model=based_on_base_model)
        self.Qs = self.load_Q_model()
        self.Q_targets = self.load_Q_model()
        self.__update_target_model__(tau=tf.constant(1.0))
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        
        # optimizer
        self.Q_optimizers = tuple(tf.optimizers.Adam(learning_rate=self.Q_lr, name=f'Q_{i}_optimizer')
                                  for i in range(self.num_Q))
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.policy_lr, name="policy_optimizer")
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=self.alpha_lr, name='alpha_optimizer')
    
    def load_Q_model(self, ):
        Qs = []
        for _ in range(self.num_Q):
            Q = SAC_critic(base_model_dir=self.base_model_dir,
                           load_sac_model=self.load_sac_model, sac_model_dir=self.sac_model_dir,
                           based_on_base_model=self.based_on_base_model)
            Qs.append(Q)
        return Qs
    
    def __heuristic_target_entropy__(self, action_space_size):
        ''' return target_entropy for discrete action space '''
        return -np.log(1.0 / action_space_size) * 0.98
    
    def actions_and_log_probs(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.policy(state)
        max_probability_action = np.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(action_probabilities, )
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = np.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action
    
    @tf.function(experimental_relax_shapes=True)
    def compute_Q_targets(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        entropy_scale = tf.convert_to_tensor(self.alpha)
        reward_scale = tf.convert_to_tensor(self.reward_scale)
        discount = tf.convert_to_tensor(self.discount_rate)
        
        next_actions, (next_pis, next_log_pis), _ = self.actions_and_log_probs(next_observations)
        next_Qs_values = tuple(Q(next_observations, next_actions) for Q in self.Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)
        next_Q_values = next_pis * (next_Q_values - entropy_scale * next_log_pis)
        next_Q_values = next_Q_values.sum(dim=1, keepdims=True)
        
        terminals = tf.cast(terminals, next_Q_values.dtype)
        Q_targets = reward_scale * rewards + discount * (1.0 - terminals) * next_Q_values
        
        return tf.stop_gradient(Q_targets)
    
    @tf.function(experimental_relax_shapes=True)
    def __update_critic__(self, batch):
        '''
        Update the Q-function.
        See Equations (5, 6) in [1], for further information of the Q-function update rule.
        '''
        
        Q_targets = self.compute_Q_targets(batch)
        
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        
        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self.Qs, self.Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q(observations, actions)
                Q_losses = 0.5 * (tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))
                Q_loss = tf.nn.compute_average_loss(Q_losses)
            
            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)
        
        return Qs_values, Qs_losses
    
    @tf.function(experimental_relax_shapes=True)
    def __update_actor__(self, batch):
        '''
        Update the policy.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        '''
        
        observations = batch['observations']
        entropy_scale = tf.convert_to_tensor(self.alpha)
        
        with tf.GradientTape() as tape:
            actions, (pis, log_pis), _ = self.actions_and_log_probs(observations)
            
            Qs_log_targets = tuple(Q(observations, actions) for Q in self.Qs)
            Q_log_targets = tf.reduce_mean(Qs_log_targets, axis=0)  # TODO: mean() or min()
            
            policy_losses = entropy_scale * log_pis - Q_log_targets
            policy_losses = (pis * policy_losses).sum(dim=1)
            policy_loss = tf.nn.compute_average_loss(policy_losses)
        
        policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))
        
        return policy_losses
    
    @tf.function(experimental_relax_shapes=True)
    def __update_alpha__(self, batch):
        observations = batch['observations']
        actions, (pis, log_pis), _ = self.actions_and_log_probs(observations)
        
        with tf.GradientTape() as tape:
            alpha_losses = pis * (-self.alpha * tf.stop_gradient(log_pis + self.target_entropy))
            alpha_losses = (pis * alpha_losses).sum(dim=1)
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
            # NOTE(hartikainen): It's important that we take the average here, \
            # otherwise we end up effectively having `batch_size` times too large learning rate.
        
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        return alpha_losses
    
    @tf.function(experimental_relax_shapes=True)
    def __update_target_model__(self, tau=None):
        if self.softUpdate:
            tau = self.tau if tau is None else tau
        else:
            tau = 1.0
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            for source_weight, target_weight in zip(Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
    
    @tf.function(experimental_relax_shapes=True)
    def update(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        Qs_values, Qs_losses = self.__update_critic__(batch)
        policy_losses = self.__update_actor__(batch)
        alpha_losses = self.__update_alpha__(batch)
    
    def _do_training(self, iteration, batch):
        training_diagnostics = self.update(batch)
        
        if iteration % self.target_update_interval == 0:
            # Run target ops here.
            self.__update_target_model__(tau=tf.constant(self.tau))
        
        return training_diagnostics
    
    def save_model(self, model_path=None, ):
        '''
        save the RL model. If ddqn, save target_model, else save model.
        :param model_path:
        :return:
        '''
        model_path = self.sac_model_dir if (model_path is None) else os.path.join(model_path, 'classifier')
        # self.model.save(model_path)
        # TODO: save model
    
    def remember(self, state, action, reward, state_, done, ):
        '''
        save the experience to memory buffer.
        '''
        
        # extract feature to remember
        state = self.feature_extractor.predict(np.array([state]))[0]
        if state_ is not None:
            state_ = self.feature_extractor.predict(np.array([state_]))[0]
        experience = state, action, reward, state_, done
        self.memory.append(experience)
    
    def remember_batch(self, batch_experience, useDiscount=True):
        '''
        save a batch of experience to memory buffer.
        if discount: apply discount to the reward.
        '''
        if useDiscount:
            for i in range(len(batch_experience) - 2, -1, -1):
                batch_experience[i][2] += self.discount_rate * batch_experience[i + 1][2]
        for experience in batch_experience:
            self.remember(*experience)
    
    def learn_sample(self, state, action, reward, state_, done, ):
        self.learn_per_batch([state], [action], [reward], [state_], [done], )
    
    def learn_per_batch(self, state, action, reward, state_, done, ):
        '''
        optimize the model for one batch
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param done:
        :return:
        '''
        state = np.array(state)
        action = np.array(action, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        state_ = np.array(state_)
        done = np.array(done, dtype=np.bool)
        
        target = self.model.predict(state)  # predict Q for starting state with the main network
        target_old = np.array(target)
        target_next = self.model.predict(state_)  # predict best action in ending state with the main network
        if self.ddqn:
            target_value = self.target_model.predict(state_)  # predict Q for ending state with the target network
        
        for i in range(len(done)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    a = np.argmax(target_next[i])  # current Q Network selects the action
                    target[i][action[i]] = reward[i] + self.discount_rate * (
                        target_value[i][a])  # target Q Network to evaluate
                else:  # Standard - DQN ---- DQN chooses the max Q value among next actions
                    target[i][action[i]] = reward[i] + self.discount_rate * (np.amax(target_next[i]))
        
        self.model.fit(state, target, batch_size=min(len(done), self.batch_size), verbose=2)
        
        return target_old, target
    
    def replay(self, ):
        '''
        experience replay and learn on a batch
        '''
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # minibatch = np.array(minibatch, dtype=object)
        # state, action, reward, next_state, done = minibatch.T
        # state, next_state = np.concatenate(state), np.concatenate(next_state)
        # action = np.asarray(action, dtype=np.int32)
        # reward = np.asarray(reward, dtype=np.float64)
        # done = np.asarray(done, dtype=np.bool)
        # state, action, reward, state_, done = list(zip(*minibatch))
        # for i, i_state_ in enumerate(state_):
        #     if i_state_ is None:
        #         state_[i] = state[i]
        state, action, reward, state_, done = [], [], [], [], []
        for i_state, i_action, i_reward, i_state_, i_done in minibatch:
            state.append(i_state), action.append(i_action), reward.append(i_reward), done.append(i_done)
            state_.append(i_state_ if (i_state_ is not None) else i_state)
        
        target_old, target = self.learn_per_batch(state, action, reward, state_, done)
    
    def learn(self, learnTimes=None):
        learnTimes = self.learnTimes if learnTimes is None else learnTimes
        for _ in range(learnTimes):
            self.replay()
    
    def act(self, state, **kwargs):
        '''
        return the action and explore_prob based on eps_decay
        :param state:
        :return:
        '''
        action_prob = self.predict(state)
        # max_prob_action = np.argmax(action_prob, dim=-1)
        action_distribution = tfp.distributions.Categorical(probs=action_prob, )
        action = action_distribution.sample()  # .cpu()
        # log_action_prob = np.log(action_prob + EPS)
        return action, action_prob[int(action)]  # (action_prob, log_action_prob)  # , max_prob_action
    
    def predict(self, state, ):
        '''
        produce the output of the given state with model.
        '''
        state = np.array([state])
        state = self.feature_extractor.predict(state)
        
        return self.policy.predict(state)
    
    # @tf.function(experimental_relax_shapes=True)
    # def td_targets(self, rewards, discounts, next_values):
    #     return rewards + discounts * next_values


if __name__ == '__main__':
    print('Hello World!')
    
    agent = SACAgent(based_on_base_model=False)
    
    print('Brand-new World!')
