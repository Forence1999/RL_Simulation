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
from agent_d3qn import DQNAgent
from env import MAP_ENV
from lib import utils


class RL_game():
    def __init__(self, ):
        super(RL_game, self).__init__()
        # -------------------------------- D3QN agent parameters ------------------------------------#
        self.num_action = 8
        self.reward_discount_rate = 0.95
        self.lr = 0.00001
        self.ddqn = True
        self.dueling = True
        self.softUpdate = True
        self.softUpdate_tau = 0.1
        self.usePER = False
        self.batch_size = 64
        self.memory_size = 1024
        self.episodes = 50
        self.eps_decay = True
        self.ini_eps = 1.0
        self.min_eps = 0.01
        self.eps_decay_rate = 0.999
        self.base_model_dir = '../model/base_model'
        self.d3qn_model_dir = '../model/d3qn_model'
        self.load_d3qn_model = False
        self.model_name = self.__gen_name__()
        print('-' * 20, 'Model Name:', self.model_name, '-' * 20, )
        self.agent = DQNAgent(num_action=self.num_action, reward_discount_rate=self.reward_discount_rate, lr=self.lr,
                              ddqn=self.ddqn, dueling=self.dueling, softUpdate=self.softUpdate,
                              softUpdate_tau=self.softUpdate_tau,
                              usePER=self.usePER, batch_size=self.batch_size, memory_size=self.memory_size,
                              eps_decay=self.eps_decay, ini_eps=self.ini_eps, min_eps=self.min_eps,
                              eps_decay_rate=self.eps_decay_rate,
                              base_model_dir=self.base_model_dir, d3qn_model_dir=self.d3qn_model_dir,
                              load_d3qn_model=self.load_d3qn_model, d3qn_model_name=self.model_name, )
        self.agent.compile()
        # ---------------------------------------------------------------------------------------#
        self.print_interval = 10
        self.max_episode_steps = 50  # 一个episode最多探索多少步，超过则强行终止。
        self.num_update_eposide = 10
        self.num_smooth_reward = 20
        self.ds_path = '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
        self.env = MAP_ENV(ds_path=self.ds_path)
    
    def __gen_name__(self, ):
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
    
    def plot_and_save_rewards(self, reward, ):
        img_path = '../model/reward_res/curve' + self.model_name + '.jpg'  # TODO: 修改数据和图像保存目录
        title = 'Episode-wise training reward - ' + self.model_name
        reward = np.array(reward)
        ave_reward = np.convolve(np.ones((self.num_smooth_reward,)) / self.num_smooth_reward,
                                 reward, mode='valid')  # 20步移动平均
        curve_name = ['Training reward', 'Ave_training reward', ]
        curve_data = [reward, ave_reward]
        color = ['r', 'g']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5,
                         show=False)
        np.savez('../model/reward_res/data' + self.model_name + '.npz', data=reward)
    
    def play(self, ):
        total_step = 0
        episode_reward = []
        for episode_idx in range(self.episodes):
            e_reward = []
            experience_ls = []
            state, done = self.env.reset()
            num_step = 0
            while not done:
                num_step += 1
                total_step += 1
                action, exp_pro = self.agent.act(state, decay_step=total_step)
                state_, reward, done, info = self.env.step(action)
                experience_ls.append([state, action, reward, state_, done])
                state = state_
                e_reward.append(reward)
                
                # if done:
                #     print('Done! Saving model...')
                #     self.agent.save_model()
                
                if num_step >= self.max_episode_steps:
                    break
            episode_reward.append(np.sum(e_reward))
            self.agent.remember_batch(batch_experience=experience_ls, useDiscount=True)
            self.agent.replay()
            
            print('episode: ', episode_idx, '\t', 'done:', done, '\t', 'steps:', num_step, '\t',
                  'crt_reward: ', np.around(episode_reward[-1], 3), '\t',
                  'avg_reward_20: ', np.around(
                    np.mean(episode_reward[-self.num_smooth_reward if episode_idx >= self.num_smooth_reward else 0:]),
                    2), )
            if (episode_idx + 1) % self.num_update_eposide == 0:
                self.agent.update_target_model()
                self.plot_and_save_rewards(episode_reward)
        self.plot_and_save_rewards(episode_reward)


if __name__ == '__main__':
    print('Hello World!')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # K.set_image_data_format('channels_first')
    
    game = RL_game()
    game.play()
    
    print('Brand-new World!')
