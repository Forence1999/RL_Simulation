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
import numpy as np
from agent_d3qn import DQNAgent
from env import MAP_ENV
import tensorflow as tf
from lib.utils import plot_curve


class RL_game():
    def __init__(self, num_action=8):
        super(RL_game, self).__init__()
        self.ddqn = True
        self.Soft_Update = True
        self.dueling = True
        self.epsilon_greedy = True
        self.USE_PER = False
        self.episodes = 500
        self.print_interval = 10
        self.max_episode_steps = 50  # 一个episode最多探索多少步，超过则强行终止。
        self.num_update_eposide = 10
        self.lr = 0.00001
        self.num_smooth_reward = 20
        
        self.name = self.__gen_name__()
        print('-' * 20, 'Model Name:', self.name, '-' * 20, )
        self.ds_path = '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
        self.env = MAP_ENV(ds_path=self.ds_path)
        self.agent = DQNAgent(num_action=num_action, ddqn=self.ddqn, Soft_Update=self.Soft_Update, dueling=self.dueling,
                              epsilon_greedy=self.epsilon_greedy, USE_PER=self.USE_PER, lr=self.lr)
        self.agent.compile()
    
    def __gen_name__(self):
        dueling = 'Dueling_' if self.dueling else ''
        dqn = 'DDQN' if self.ddqn else 'DQN'
        softupdate = '_SoftUpdate' if self.Soft_Update else ''
        greedy = '_Greedy' if self.epsilon_greedy else ''
        PER = '_PER' if self.USE_PER else ''
        lr = '_lr_' + str(self.lr)
        name = dueling + dqn + softupdate + greedy + PER + lr
        
        return name
    
    def plot_and_save_rewards(self, reward, ):
        img_path = './' + self.name + '.jpg'  # TODO: 修改数据和图像保存目录
        title = 'Episode-wise training reward - ' + self.name
        reward = np.array(reward)
        ave_reward = np.convolve(np.ones((self.num_smooth_reward,)) / self.num_smooth_reward,
                                 reward, mode='valid')  # 20步移动平均
        curve_name = ['Training reward', 'Ave_training reward', ]
        curve_data = [reward, ave_reward]
        color = ['r', 'g']
        plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=img_path, linewidth=0.5,
                   show=False)
        np.savez('./' + self.name + '.npz', data=reward)
    
    def play(self):
        total_step = 0
        episode_reward = []
        for episode_idx in range(self.episodes):
            e_reward = []
            state, done = self.env.reset()
            num_step = 0
            while not done:
                num_step += 1
                total_step += 1
                action, exp_pro = self.agent.act(state, decay_step=total_step)
                state_, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, state_, done)
                state = state_
                e_reward.append(reward)
                
                if done:
                    print('Done! Saving model...')
                    self.agent.save_model()
                self.agent.replay()
                
                if num_step >= self.max_episode_steps:
                    break
            episode_reward.append(np.sum(e_reward))
            print('episode: ', episode_idx, '\n',
                  'crt_reward: ', np.around(episode_reward[-1], 3), '\n',
                  'avg_reward_20: ', np.around(
                    np.mean(episode_reward[-self.num_smooth_reward if episode_idx >= self.num_smooth_reward else 0:]),
                    2), '\n', )
            if (episode_idx + 1) % self.num_update_eposide == 0:
                self.agent.update_target_model()
                self.plot_and_save_rewards(episode_reward)
        self.plot_and_save_rewards(episode_reward)


if __name__ == '__main__':
    print('Hello World!')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # K.set_image_data_format('channels_first')
    
    game = RL_game()
    game.play()
    
    print('Brand-new World!')
