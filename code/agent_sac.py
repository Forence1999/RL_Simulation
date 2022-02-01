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


class SAC(object):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """
    
    def __init__(self, num_action=8, reward_scale=1.0, reward_discount_rate=0.99,
                 policy_lr=3e-4, Q_lr=3e-4, alpha_lr=3e-4,
                 batch_size=8, memory_size=64,
                 softUpdate=True, softUpdate_tau=5e-3, target_update_interval=1,
                 base_model_dir='../model/base_model', sac_model_dir='../model/d3qn_model',
                 load_sac_model=True, sac_model_name=None,
                 **kwargs, ):
        super(SAC, self).__init__(**kwargs)
        assert (not softUpdate), 'softUpdate must be True for SAC'
        
        self.name = self.__gen_name__() if (sac_model_name is None) else sac_model_name
        
        # space
        self.num_action = num_action
        # reward
        self.reward_scale = reward_scale
        self.discount_rate = reward_discount_rate
        # optimize
        self.softUpdate = softUpdate  # use soft parameter update
        self.tau = softUpdate_tau  # target network soft update hyperparameter
        self.policy_lr = policy_lr
        self.Q_lr = Q_lr
        self.alpha_lr = alpha_lr
        self.target_update_interval = target_update_interval
        self.target_entropy = self.heuristic_target_entropy(self.num_action)
        # memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        # model
        self.load_sac_model = load_sac_model
        self.base_model_dir = base_model_dir
        self.sac_model_dir = os.path.join(sac_model_dir, self.name)
        os.makedirs(self.sac_model_dir, exist_ok=True)
        
        # optimizer
        self.policy = self.load_p_model()
        self.Qs = self.load_q_model()
        self.Q_targets = tuple(deepcopy(Q) for Q in self.Qs)
        self.update_target(tau=tf.constant(1.0))
        
        self.Q_optimizers = tuple(tf.optimizers.Adam(learning_rate=self.Q_lr, name=f'Q_{i}_optimizer')
                                  for i, Q in enumerate(self.Qs))
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.policy_lr, name="policy_optimizer")
        
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optimizer = tf.optimizers.Adam(self.alpha_lr, name='alpha_optimizer')
    
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
    
    def load_q_model(self, ):
        return [None] * 2
    
    def load_p_model(self, ):
        pass
    
    def heuristic_target_entropy(self, action_space_size):
        ''' return target_entropy for discrete action space '''
        return -np.log(1.0 / action_space_size) * 0.98
    
    @tf.function(experimental_relax_shapes=True)
    def td_targets(self, rewards, discounts, next_values):
        return rewards + discounts * next_values
    
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
        Q_targets = self.td_targets(rewards=reward_scale * rewards,
                                    discounts=discount,
                                    next_values=(1.0 - terminals) * next_Q_values)
        
        return tf.stop_gradient(Q_targets)
    
    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
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
    def _update_actor(self, batch):
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
    def _update_alpha(self, batch):
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
    def _update_target(self, tau):
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            for source_weight, target_weight in zip(Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)
    
    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        Qs_values, Qs_losses = self.update_critic(batch)
        policy_losses = self.update_actor(batch)
        alpha_losses = self.update_alpha(batch)
    
    def _do_training(self, iteration, batch):
        training_diagnostics = self.do_updates(batch)
        
        if iteration % self.target_update_interval == 0:
            # Run target ops here.
            self.update_target(tau=tf.constant(self.tau))
        
        return training_diagnostics
    
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
                target_weight = target_weight * (1 - self.tau) + weight * self.tau
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
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        state, action, reward, state_, done = [], [], [], [], []
        for i_state, i_action, i_reward, i_state_, i_done in minibatch:
            state.append(i_state), action.append(i_action), reward.append(i_reward), done.append(i_done)
            state_.append(i_state_ if (i_state_ is not None) else i_state)
        target_old, target = self.learn_per_batch(state, action, reward, state_, done)
    
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
        try:
            target_next = self.model.predict(state_)  # predict best action in ending state with the main network
        except:
            pass
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
            explore_prob = self.min_eps + (self.ini_eps - self.min_eps) * self.eps_decay_rate ** decay_step
        else:
            explore_prob = self.min_eps
        
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
    print('Hello World!')
    
    agent = SAC()
    
    print('Brand-new World!')
