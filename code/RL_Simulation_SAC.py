# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: main.py
# @Time: 2021/10/28/14:36
# @Software: PyCharm
import os
import sys
import time
import numpy as np
# from agent_d3qn import DQNAgent
from agent.agent_sac import SACAgent
from environment.tree_env import MAP_ENV
from lib import utils


class RL_game():
    def __init__(self, agent_learn=True, episodes=1500,
                 reward_discount_rate=0.75,
                 policy_lr=4e-4, Q_lr=4e-4, alpha_lr=4e-4,
                 learnTimes=8, batch_size=32, memory_size=1024,
                 num_update_episode=1, softUpdate=True, softUpdate_tau=0.01,
                 base_model_dir='../model/base_model_fullData_woBN', agent_model_dir='../model/sac_model',
                 load_sac_model=True, based_on_base_model=False, model_name=None,
                 max_episode_steps=30, **kwargs):
        super(RL_game, self).__init__()
        self.AGENT_CLASS = 'SAC'
        self.useMask = True
        # self.agent_learn = True
        self.agent_learn = agent_learn
        self.print_interval = 10
        self.max_episode_steps = max_episode_steps  # 一个episode最多探索多少步，超过则强行终止。
        # self.num_update_episode = 10  # update target model and reward graph & data
        self.num_update_episode = num_update_episode  # update target model and reward graph & data
        self.num_smooth_reward = 20
        self.num_action = 8
        self.episodes = episodes
        self.num_save_episode = 1000
        self.num_plot_episode = 100
        ''''''
        # -------------------------------- SAC agent parameters ------------------------------------#
        self.num_Q = 2
        self.reward_scale = 1.0
        self.reward_discount_rate = reward_discount_rate
        self.policy_lr = policy_lr
        self.Q_lr = Q_lr
        self.alpha_lr = alpha_lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.learnTimes = learnTimes
        self.softUpdate = softUpdate
        self.softUpdate_tau = softUpdate_tau
        self.base_model_dir = base_model_dir
        self.agent_model_dir = agent_model_dir
        self.load_sac_model = load_sac_model
        self.based_on_base_model = based_on_base_model
        self.model_name = self.__gen_name__() if model_name is None else model_name
        print('-' * 20, 'Model Name:', self.model_name, '-' * 20, )
        
        self.agent = SACAgent(num_Q=self.num_Q, num_action=self.num_action, reward_scale=self.reward_scale,
                              reward_discount_rate=self.reward_discount_rate,
                              policy_lr=self.policy_lr, Q_lr=self.Q_lr, alpha_lr=self.alpha_lr,
                              batch_size=self.batch_size, memory_size=self.memory_size, learnTimes=self.learnTimes,
                              softUpdate=self.softUpdate, softUpdate_tau=self.softUpdate_tau,
                              base_model_dir=self.base_model_dir, sac_model_dir=self.agent_model_dir,
                              load_sac_model=self.load_sac_model, sac_model_name=self.model_name,
                              based_on_base_model=self.based_on_base_model, )
        # ---------------------------------------------------------------------------------------#
        
        # -------------------------------- Environment parameters ------------------------------------#
        self.ds_path = os.path.abspath(
            '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5')
        self.env = MAP_ENV(ds_path=self.ds_path)
        # ---------------------------------------------------------------------------------------#
        
        self.save_config()
    
    def __gen_name__(self, ):
        agent_name = self.AGENT_CLASS
        rwd = '-'.join(('rwd', str(self.reward_scale), str(self.reward_discount_rate)))
        lr = '-'.join(('lr', 'p', str(self.policy_lr), 'Q', str(self.Q_lr), 'a', str(self.alpha_lr)))
        mem = '-'.join(('mom', str(self.memory_size), str(self.batch_size)))
        up = '-'.join(('up', str(self.num_update_episode), str(self.learnTimes), 'tau', str(self.softUpdate_tau),))
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        
        name = '_'.join((time_stamp, 'Tree', agent_name, lr, up, mem, rwd,)).replace('__', '_')
        print('-' * 20, 'Model Name:', name, '-' * 20, )
        
        return name
    
    def save_config(self, ):
        config_path = os.path.join(self.agent_model_dir, self.model_name, 'summary', 'config.json')
        config = {}
        for key in list(self.__dict__.keys()):
            if isinstance(self.__dict__[key], (str, int, float, bool)):
                config[key] = self.__dict__[key]
        
        utils.json_writer(data=config, path=config_path)
    
    def plot_and_save_rewards(self, reward, ):
        img_path = os.path.join(self.agent_model_dir, self.model_name, 'summary', 'curve.jpg')
        title = 'Training reward - ' + self.model_name
        reward = np.array(reward)
        ave_reward = np.convolve(np.ones((self.num_smooth_reward,)) / self.num_smooth_reward,
                                 reward, mode='valid')  # 20步移动平均
        curve_name = ['Training reward', 'Ave_training reward', ]
        curve_data = [reward, ave_reward]
        color = ['r', 'g']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5,
                         show=False)
        
        data_path = os.path.join(self.agent_model_dir, self.model_name, 'summary', 'reward.npz')
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez(data_path, data=reward)
    
    def play(self, ):
        total_step = 0
        episode_reward = []
        for episode_idx in range(self.episodes):
            e_reward = []
            experience_ls = []
            state, done = self.env.reset()
            num_step = 0
            act_info = None
            while not done:
                num_step += 1
                total_step += 1
                action, act_info = self.agent.act(state, decay_step=total_step, useMask=self.useMask)
                state_, reward, done, info = self.env.step(action)
                experience_ls.append([state, action, reward, state_, done])
                state = state_
                e_reward.append(reward)
                
                if num_step >= self.max_episode_steps:
                    break
            episode_reward.append(np.sum(e_reward))
            if self.agent_learn:
                self.agent.remember_batch(batch_experience=experience_ls, useDiscount=True, useMask=self.useMask)
                self.agent.learn(useMask=self.useMask)
            
            print('episode:', episode_idx, '\t',
                  'done:', done, '\t',
                  'steps:', num_step, '\t',
                  'crt_reward:', np.around(episode_reward[-1], 3), '\t',
                  f'avg_reward_{self.num_smooth_reward}:',
                  np.around(
                      np.mean(episode_reward[-self.num_smooth_reward if episode_idx >= self.num_smooth_reward else 0:]),
                      2), '\t',
                  'act_info:', act_info)
            if (episode_idx + 1) % self.num_update_episode == 0:
                self.agent.update_target_model()
            if (episode_idx + 1) % self.num_plot_episode == 0:
                self.plot_and_save_rewards(episode_reward)
            if (episode_idx + 1) % self.num_save_episode == 0:
                print('Skipping saving model...')
                # print('Done! Saving model...')
                # self.agent.save_model()
        
        self.plot_and_save_rewards(episode_reward)


if __name__ == '__main__':
    print('Hello World!')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # K.set_image_data_format('channels_first')
    import ast
    import argparse
    
    parser = argparse.ArgumentParser(description='parameters for D3QN agent.')
    
    parser.add_argument('--agent_learn', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--max_episode_steps', type=int, default=70, help='')
    parser.add_argument('--episodes', type=int, default=4000, help='')
    parser.add_argument('--policy_lr', type=float, default=3e-4, help='')
    parser.add_argument('--Q_lr', type=float, default=3e-4, help='')
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='')
    parser.add_argument('--reward_discount_rate', type=float, default=0.75, help='')
    
    parser.add_argument('--learnTimes', type=int, default=8, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--memory_size', type=int, default=4096, help='')
    
    parser.add_argument('--softUpdate', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--num_update_episode', type=int, default=1, help='')
    parser.add_argument('--softUpdate_tau', type=float, default=0.01, help='')
    
    parser.add_argument('--base_model_dir', type=str, default='../model/base_model_fullData_wBN', help='')
    parser.add_argument('--based_on_base_model', type=ast.literal_eval, default=True, help='')
    
    parser.add_argument('--agent_model_dir', type=str, default='../model/sac_model', help='')
    # parser.add_argument('--model_name', type=str,
    #                     default='20220214-004841_SAC_lr-p-0.0003-Q-0.0003-a-0.0003_up-1-8-tau-0.01_mom-1024-32_rwd-1.0-0.75',
    #                     help='')
    parser.add_argument('--load_sac_model', type=ast.literal_eval, default=False, help='')
    
    args = vars(parser.parse_args())
    print('RL config:', args)
    
    game = RL_game(**args)
    game.play()
    
    print('Brand-new World!')
