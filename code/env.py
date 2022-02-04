# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: env.py
# @Time: 2021/10/31/10:43
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
from map import Map_graph
import pickle
from lib import utils


class MAP_ENV(object):
    def __init__(self, ds_path, ):
        super(MAP_ENV, self).__init__()
        self.ds_path = ds_path
        self.map = Map_graph(ds_path=self.ds_path)
        self.num_actions = 8
        
        self.src_id = None
        self.wk_id = None
        self.abs_doa = None
        self.done = None
    
    def reset(self, ):
        '''
        for the state of an epoch, reset all the state variables.
        :return:
        '''
        self.src_id = self.map.random_id()
        while True:
            self.wk_id = self.map.random_id()
            if self.src_id != self.wk_id:
                break
        self.abs_doa = self.map.random_doa()
        state = self.get_state(src_id=self.src_id, wk_id=self.wk_id, abs_doa=self.abs_doa)
        self.done = False
        
        return state, self.done
    
    def get_relative_doa(self, src_id, wk_id, abs_doa, ):
        '''
        get the relative doa of src_id relative to walker (abs_doa and wk_id)
        :param src_id:
        :param wk_id:
        :param abs_doa:
        :return:
        '''
        src_2_wk_doa = self.map.find_relative_direction(src_id=src_id, wk_id=wk_id)
        rela_doa = (src_2_wk_doa - abs_doa - 2 + 16) % 8
        return rela_doa
    
    def load_preprocess_state(self, state_path):
        '''
        load and preprocess state.
        :param state:
        :return:
        '''
        x = np.load(state_path)['data'][:, :, 5:].transpose((1, 2, 0))
        # x = x.transpose((1, 0, 2))
        
        # log
        x_sign = np.sign(x)
        x_abs = np.abs(x)
        x_log = np.log(x_abs + 1)
        x = x_sign * x_log
        # norm
        # x = (x - np.mean(x)) / np.std(x)
        # x = x / np.std(x)
        
        return x
    
    def get_state(self, src_id, wk_id, abs_doa, ):
        '''
        基于 src_id 和 wk_id 返回 walker 目前所能接收到的状态。 # 接受相邻三个方向的数据
        若声源相对于小车的数据不存在，则保持小车的位置不变，找到能够覆盖小车数据的离声源最近的次声源，并返回其对 walker 位置的数据。
        :param src_id:
        :param wk_id:
        :param abs_doa: The direction which the walker is facing relative to the absolute coordinate system
        :return:
        '''
        wk_coord = self.map.get_coordinate(wk_id)
        wk_basename = '_'.join(['walker', ] + list(map(str, wk_coord)) + ['1', ])
        data_path = self.map.find_shortest_data_path(src_id=src_id, wk_id=wk_id, )
        sub_src_id = data_path[-2]
        sub_src_coord = self.map.get_coordinate(sub_src_id)
        sub_src_basename = '_'.join(['src', ] + list(map(str, sub_src_coord)))
        rela_doa = self.get_relative_doa(src_id=sub_src_id, wk_id=wk_id, abs_doa=abs_doa)
        # print('sub_src_key:', sub_src_key, 'wk_key:', wk_key)
        
        state_path_ls = None
        for i in range(2):
            doa_ls = list({(rela_doa + i + self.num_actions) % self.num_actions,
                           (rela_doa - i + self.num_actions) % self.num_actions, })
            random.shuffle(doa_ls)
            for j in doa_ls:
                str_doa = str(j * 45)
                doa_data_path = os.path.join(self.ds_path, sub_src_basename, wk_basename, str_doa, )
                if os.path.exists(doa_data_path):
                    state_path_ls = utils.get_files_by_suffix(root=doa_data_path, suffix='.npz')
                    break
            if state_path_ls is not None:
                break
        
        if state_path_ls is None:
            print('data_path:', data_path, )
            print('src_id:', src_id, )
            print('sub_src_id:', sub_src_id, 'sub_src_basename:', sub_src_basename, )
            print('wk_id:', wk_id, 'wk_basename:', wk_basename, )
            print('abs_doa:', abs_doa)
            print('rela_doa:', rela_doa)
            raise ValueError('src_walker data does not exist')
        state_path = random.sample(state_path_ls, 1)[0]
        state = self.load_preprocess_state(state_path=state_path)
        return state
    
    def next_pose(self, action, ):
        '''
        the next position and abs_doa if walker takes the action.
        If the adjacent 3 direction is not available, walker will staty still.
        :param action:
        :return:
        '''
        abs_action = self.get_abs_action(action)
        
        next_id, next_abs_doa = None, None
        for i in range(2):
            doa_ls = list({(abs_action + i + self.num_actions) % self.num_actions,
                           (abs_action - i + self.num_actions) % self.num_actions, })
            neighbors = self.map.nodes[self.wk_id].get_neighbor()
            doa_node_pair = [(doa, node) for doa, node in enumerate(neighbors) if
                             ((doa in doa_ls) and (node is not None))]
            # id_ls = np.asarray(id_ls)
            # id_ls = id_ls[id_ls != None]
            if len(doa_node_pair) > 0:
                next_abs_doa, next_id = random.sample(doa_node_pair, 1)[0]
                break
        
        if next_id is None:
            return False
        else:
            self.wk_id, self.abs_doa = next_id, (next_abs_doa - 2 + 8) % 8
            return True
    
    def get_abs_action(self, action, ):
        '''
        return the direction walker will face if it takes the action.
        :param action:
        :return:
        '''
        return (action + self.abs_doa + 2) % self.num_actions
    
    def step(self, action, ):
        '''
        Ask walker to take the action and update the relevant state variables.
        :param action:
        :return:
        '''
        doAction = self.next_pose(action=action)
        
        # setting rewards
        reward = -0.05
        if not doAction:
            reward -= 0.1
        
        if self.src_id == self.wk_id:
            done = True
            reward += 1.
            state_ = None
        else:
            done = False
            state_ = self.get_state(src_id=self.src_id, wk_id=self.wk_id, abs_doa=self.abs_doa)
        
        info = None
        return state_, reward, done, info
    
    def render(self, ):
        '''
        display the env.
        :return:
        '''
        pass


if __name__ == '__main__':
    print('Hello World!')
    
    ds_path = '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
    map_env = MAP_ENV(ds_path=ds_path)
    map_env.get_state(1, 20, 1)
    print('Brand-new World!')
