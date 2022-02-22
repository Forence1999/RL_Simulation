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
from agent.agent_d3qn import DQNAgent
from environment.tree_env import MAP_ENV
from lib import utils


class RL_game():
    def __init__(self, agent_learn=True, reward_discount_rate=0.75, lr=0.0001, softUpdate_tau=0.1,
                 batch_size=64, learnTimes=1, episodes=500, eps_decay=False, min_eps=0.1,
                 base_model_dir='../model/base_model', load_d3qn_model=False, based_on_base_model=True,
                 num_update_episode=1, d3qn_model_dir=None, model_name=None, memory_size=1024, **kwargs):
        super(RL_game, self).__init__()
        self.AGENT_CLASS = 'D3QN'
        self.useMask = True
        # self.agent_learn = True
        self.agent_learn = agent_learn
        self.print_interval = 10
        self.max_episode_steps = 50  # 一个episode最多探索多少步，超过则强行终止。
        # self.num_update_episode = 10  # update target model and reward graph & data
        self.num_update_episode = num_update_episode  # update target model and reward graph & data
        self.num_save_episode = 100
        self.num_smooth_reward = 20
        self.num_plot_episode = 100
        
        # -------------------------------- D3QN agent parameters ------------------------------------#
        self.num_action = 8
        # self.reward_discount_rate = 0.95  # [0.8, 0.95]
        self.reward_discount_rate = reward_discount_rate  # [0.8, 0.95]
        # self.lr = 0.00001  # [1e-5, 3e-4]
        self.lr = lr  # [1e-5, 3e-4]
        self.ddqn = True
        self.dueling = True
        self.softUpdate = True
        # self.softUpdate_tau = 0.1
        self.softUpdate_tau = softUpdate_tau
        # self.learnTimes = 1
        self.learnTimes = learnTimes
        self.usePER = False
        # self.batch_size = 64
        self.batch_size = batch_size
        # self.memory_size = 1024
        self.memory_size = memory_size
        # self.episodes = 500
        self.episodes = episodes
        # self.eps_decay = False
        self.eps_decay = eps_decay
        self.ini_eps = 1.0
        # self.min_eps = 0.1
        self.min_eps = min_eps
        self.eps_decay_rate = 0.999
        # self.base_model_dir = '../model/base_model'
        self.base_model_dir = base_model_dir
        self.d3qn_model_dir = '../model/d3qn_model' if d3qn_model_dir is None else d3qn_model_dir
        # self.load_d3qn_model = False
        self.load_d3qn_model = load_d3qn_model
        # self.based_on_base_model = True
        self.based_on_base_model = based_on_base_model
        self.model_name = self.__gen_name__() if model_name is None else model_name
        print('-' * 20, 'Model Name:', self.model_name, '-' * 20, )
        
        self.agent = DQNAgent(num_action=self.num_action, reward_discount_rate=self.reward_discount_rate, lr=self.lr,
                              ddqn=self.ddqn, dueling=self.dueling, softUpdate=self.softUpdate,
                              softUpdate_tau=self.softUpdate_tau, learnTimes=self.learnTimes,
                              usePER=self.usePER, batch_size=self.batch_size, memory_size=self.memory_size,
                              eps_decay=self.eps_decay, ini_eps=self.ini_eps, min_eps=self.min_eps,
                              eps_decay_rate=self.eps_decay_rate,
                              base_model_dir=self.base_model_dir, d3qn_model_dir=self.d3qn_model_dir,
                              load_d3qn_model=self.load_d3qn_model, based_on_base_model=self.based_on_base_model,
                              d3qn_model_name=self.model_name, )
        # ---------------------------------------------------------------------------------------#
        
        # -------------------------------- Environment parameters ------------------------------------#
        self.ds_path = os.path.abspath(
            '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5')
        self.env = MAP_ENV(ds_path=self.ds_path)
        # ---------------------------------------------------------------------------------------#
        
        self.save_config()
    
    def __gen_name__(self, ):
        dueling = 'Dueling' if self.dueling else ''
        dqn = 'DDQN' if self.ddqn else 'DQN'
        softUpdate = 'softUpdate' if self.softUpdate else ''
        eps_decay = 'epsDecay' if self.eps_decay else ''
        usePER = 'usePER' if self.usePER else ''
        lr = 'lr_' + str(self.lr)  # TODO
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        
        name = '_'.join((time_stamp, 'Tree', dueling, dqn, softUpdate, eps_decay, usePER, lr,)).replace('__', '_')
        print('-' * 20, 'Model Name:', name, '-' * 20, )
        
        return name
    
    def save_config(self, ):
        config_path = os.path.join(self.d3qn_model_dir, self.model_name, 'summary', 'config.json')
        config = {}
        for key in list(self.__dict__.keys()):
            if isinstance(self.__dict__[key], (str, int, float, bool)):
                config[key] = self.__dict__[key]
        
        utils.json_writer(data=config, path=config_path)
    
    def plot_and_save_rewards(self, reward, ):
        # img_path = '../model/reward_res/curve/' + self.model_name + '.jpg'
        img_path = os.path.join(self.d3qn_model_dir, self.model_name, 'summary', 'curve.jpg')
        title = 'Training reward - ' + self.model_name
        reward = np.array(reward)
        ave_reward = np.convolve(np.ones((self.num_smooth_reward,)) / self.num_smooth_reward,
                                 reward, mode='valid')  # 20步移动平均
        curve_name = ['Training reward', 'Ave_training reward', ]
        curve_data = [reward, ave_reward]
        color = ['r', 'g']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5,
                         show=False)
        # os.makedirs('../model/reward_res/data/', exist_ok=True)
        # np.savez('../model/reward_res/data/' + self.model_name + '.npz', data=reward)
        
        data_path = os.path.join(self.d3qn_model_dir, self.model_name, 'summary', 'reward.npz')
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez(data_path, data=reward)
    
    def play(self, ):
        print('-' * 20, 'Start playing', '-' * 20, )
        total_step = 0
        episode_rewards = []
        for episode_idx in range(self.episodes):
            step_rewards = []
            experience_ls = []
            state, done = self.env.reset()  # if useMask: state: (state, mask)
            num_step = 0
            exp_pro = None
            while not done:
                num_step += 1
                total_step += 1
                
                action, exp_pro = self.agent.act(state[0], decay_step=total_step, mask=state[1]) if self.useMask \
                    else self.agent.act(state[0], decay_step=total_step, )
                state_, reward, done, info = self.env.step(action)
                experience_ls.append([state, action, reward, state_, done])
                state = state_
                step_rewards.append(reward)
                
                if num_step >= self.max_episode_steps:
                    break
            episode_rewards.append(np.sum(step_rewards))
            if self.agent_learn:
                self.agent.remember_batch(batch_experience=experience_ls, useDiscount=True, useMask=self.useMask)
                self.agent.learn(useMask=self.useMask)
            
            print('episode:', episode_idx, '\t',
                  'done:', done, '\t',
                  'steps:', num_step, '\t',
                  'crt_reward:', np.around(episode_rewards[-1], 3), '\t',
                  f'avg_reward_{self.num_smooth_reward}:',
                  np.around(
                      np.mean(
                          episode_rewards[-self.num_smooth_reward if episode_idx >= self.num_smooth_reward else 0:]),
                      2), '\t',
                  'exp_pro:', exp_pro)
            if (episode_idx + 1) % self.num_update_episode == 0:
                self.agent.update_target_model()
            if (episode_idx + 1) % self.num_plot_episode == 0:
                self.plot_and_save_rewards(episode_rewards)
            if (episode_idx + 1) % self.num_save_episode == 0:
                # print('Skipping saving model...')
                print('Done! Saving model...')
                self.agent.save_model()
        
        self.plot_and_save_rewards(episode_rewards)


if __name__ == '__main__':
    print('Hello World!')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # K.set_image_data_format('channels_first')
    np.set_printoptions(precision=3, suppress=True)
    
    import ast
    import argparse
    
    parser = argparse.ArgumentParser(description='parameters for D3QN agent.')
    # parser.add_argument('--agent_learn', type=ast.literal_eval, default=True, help='')
    # parser.add_argument('--reward_discount_rate', type=float, default=0.9, help='')
    # parser.add_argument('--lr', type=float, default=0.00001, help='')
    # parser.add_argument('--dueling', type=ast.literal_eval, default=True, help='')
    # parser.add_argument('--softUpdate_tau', type=float, default=0.1, help='')
    # parser.add_argument('--batch_size', type=int, default=64, help='')
    # parser.add_argument('--memory_size', type=int, default=1024, help='')
    # parser.add_argument('--episodes', type=int, default=500, help='')
    # parser.add_argument('--eps_decay', type=ast.literal_eval, default=False, help='')
    # parser.add_argument('--min_eps', type=float, default=0.1, help='')
    # parser.add_argument('--base_model_dir', type=str, default='../model/base_model', help='')
    # parser.add_argument('--load_d3qn_model', type=ast.literal_eval, default=False, help='')
    # parser.add_argument('--based_on_base_model', type=ast.literal_eval, default=False, help='')
    # parser.add_argument('--num_update_episode', type=int, default=10, help='')
    
    parser.add_argument('--agent_learn', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--memory_size', type=int, default=4096, help='')
    parser.add_argument('--learnTimes', type=int, default=8, help='')
    parser.add_argument('--num_update_episode', type=int, default=1, help='')
    parser.add_argument('--softUpdate_tau', type=float, default=0.01, help='')
    parser.add_argument('--episodes', type=int, default=1500, help='')
    parser.add_argument('--reward_discount_rate', type=float, default=0.75, help='')
    
    parser.add_argument('--eps_decay', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--min_eps', type=float, default=0.1, help='')
    parser.add_argument('--base_model_dir', type=str, default='../model/base_model_fullData_wBN', help='')
    parser.add_argument('--d3qn_model_dir', type=str, default='../model/d3qn_model', help='')
    # parser.add_argument('--model_name', type=str, default='Dueling_DDQN_softUpdate__lr_0.0001_20220212-235802', help='')
    parser.add_argument('--load_d3qn_model', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--based_on_base_model', type=ast.literal_eval, default=True, help='')
    
    args = vars(parser.parse_args())
    print('RL config:', args)
    
    game = RL_game(**args)
    game.play()
    
    print('Brand-new World!')
