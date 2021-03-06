# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: env.py
# @Time: 2021/10/31/10:43
# @Software: PyCharm

import os, sys

F_dir = os.path.dirname(os.path.abspath(__file__))
FF_dir = os.path.dirname(F_dir)
sys.path.append(FF_dir)

import time
import random
import warnings
import numpy as np
from copy import deepcopy
from .tree_map import Map_graph
import pickle
from lib import utils


class MAP_ENV(object):
    def __init__(self, ds_path, ):  # Checked
        super(MAP_ENV, self).__init__()
        self.ds_path = ds_path
        self.map = Map_graph(ds_path=self.ds_path)
        self.num_actions = 8
        
        self.src_id = None
        self.wk_id = None
        self.abs_doa = None
        self.done = None
    
    def reset(self, ):  # Checked
        '''
        for the state of an epoch, reset all the state variables.
        :return:
        '''
        src_id = self.map.random_id()
        while True:
            wk_id = self.map.random_id()
            if not self.map.inSameRoom(id1=src_id, id2=wk_id):
                break
        abs_doa = self.map.random_doa()
        state = self.get_state(src_id=src_id, wk_id=wk_id, abs_doa=abs_doa)
        mask = self.get_mask(wk_id=wk_id, abs_doa=abs_doa)
        done = False
        
        self.src_id, self.wk_id, self.abs_doa, self.done = src_id, wk_id, abs_doa, done
        self.subsrc_id = None
        
        return (state, mask), self.done
    
    def get_src_doa(self, src_id, wk_id, abs_doa, ):  # Checked
        '''
        get the relative doa of src_id relative to walker (abs_doa and wk_id)
        :param src_id:
        :param wk_id:
        :param abs_doa:
        :return:
        '''
        src_2_wk_doa = self.map.find_relative_direction(src_id=src_id, wk_id=wk_id)
        src_doa = (src_2_wk_doa - abs_doa + 8) % 8
        return src_doa
    
    def load_preprocess_state(self, state_path):  # Checked
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
    
    def get_state(self, src_id, wk_id, abs_doa, ):  # Checked
        '''
        ?????? src_id ??? wk_id ?????? walker ????????????????????????????????? # ?????????????????????????????????
        ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? walker ??????????????????
        :param src_id:
        :param wk_id:
        :param abs_doa: The direction which the walker is facing relative to the absolute coordinate system
        :return:
        '''
        wk_coord = self.map.get_room_center(id=wk_id)
        wk_basename = '_'.join(['walker', ] + list(map(str, wk_coord)) + ['1', ])
        data_path = self.map.find_shortest_data_path(src_id=src_id, wk_id=wk_id, )
        sub_src_id = data_path[-2]
        self.sub_src_id = sub_src_id
        sub_src_coord = self.map.get_room_center(id=sub_src_id)
        sub_src_basename = '_'.join(['src', ] + list(map(str, sub_src_coord)))
        src_doa = self.get_src_doa(src_id=sub_src_id, wk_id=wk_id, abs_doa=abs_doa)
        
        # print('sub_src_key:', sub_src_key, 'wk_key:', wk_key)
        
        state_path_ls = None
        for i in range(2):
            doa_ls = list({(src_doa + i + self.num_actions) % self.num_actions,
                           (src_doa - i + self.num_actions) % self.num_actions, })
            random.shuffle(doa_ls)
            for j in doa_ls:
                str_doa = str(j * 45)
                doa_data_path = os.path.join(self.ds_path, sub_src_basename, wk_basename, str_doa, )
                if os.path.exists(doa_data_path):
                    state_path_ls = utils.get_files_by_suffix(root=doa_data_path, suffix='.npz')
                    self.sub_src_doa = j
                    break
            if state_path_ls is not None:
                break
        
        if state_path_ls is None:
            print('data_path:', data_path, )
            print('src_id:', src_id, 'room_id', self.map.find_room(id=src_id).id)
            print('sub_src_id:', sub_src_id, 'room_id', self.map.find_room(id=sub_src_id).id,
                  'sub_src_basename:', sub_src_basename)
            print('wk_id:', wk_id, 'room_id', self.map.find_room(id=wk_id).id, 'wk_basename:', wk_basename)
            print('abs_doa:', abs_doa)
            print('src_doa:', src_doa)
            raise ValueError('src_walker data does not exist')
        state_path = random.choice(state_path_ls)
        state = self.load_preprocess_state(state_path=state_path)
        
        return state
    
    def get_mask(self, wk_id, abs_doa, ):  # Checked
        ''' A mask represents which directions the walker can choose. '''
        crt_neighbors = self.map.get_neighbors(id=wk_id)
        mask = [i is not None for i in crt_neighbors]
        mask = np.roll(mask, shift=-abs_doa)
        
        return mask
    
    def next_pose(self, action, ):  # Checked
        '''
        the next position and abs_doa if walker takes the action.
        If the adjacent 3 direction is not available, walker will staty still.
        :param action:
        :return:
        '''
        abs_action = self.get_abs_action(action)
        
        # next_id, next_abs_doa = None, None
        # for i in range(2):
        #     doa_ls = list({(abs_action + i + self.num_actions) % self.num_actions,
        #                    (abs_action - i + self.num_actions) % self.num_actions, })
        #     neighbors = self.map.nodes[self.wk_id].get_neighbor()
        #     doa_node_pair = [(doa, node) for doa, node in enumerate(neighbors) if
        #                      ((doa in doa_ls) and (node is not None))]
        #     # id_ls = np.asarray(id_ls)
        #     # id_ls = id_ls[id_ls != None]
        #     if len(doa_node_pair) > 0:
        #         next_abs_doa, next_id = random.choice(doa_node_pair)
        #         break
        # if next_id is None:
        #     return False
        # else:
        #     self.wk_id, self.abs_doa = next_id, (next_abs_doa - 2 + 8) % 8
        #     return True
        self.wk_id = self.map.find_id_by_direction(base_id=self.wk_id, direction=abs_action)
        self.abs_doa = abs_action
    
    def get_abs_action(self, action, ):  # Checked
        '''
        return the direction walker will face if it takes the action.
        :param action:
        :return:
        '''
        return (action + self.abs_doa) % self.num_actions
    
    def generate_reward(self, action):
        abs_action = self.get_abs_action(action)
        src_doa = self.sub_src_doa
        diff_doa = 4 - abs(abs(src_doa - abs_action) - 4)
        
        if diff_doa == 0:
            reward = 0.6
        elif diff_doa == 1:
            reward = 0.4
        elif diff_doa == 2:
            reward = 0.2
        elif diff_doa == 3:
            reward = -0.2
        elif diff_doa == 4:
            reward = -0.4
        else:
            raise ValueError('diff_doa:', diff_doa)
        
        return reward
    
    def step(self, action, ):  # Checked
        '''
        Ask walker to take the action and update the relevant state variables.
        :param action:
        :return:
        '''
        
        # setting rewards
        
        reward = self.generate_reward(action)
        self.next_pose(action=action)
        
        if self.map.inSameRoom(id1=self.src_id, id2=self.wk_id):
            done = True
            state_ = None
            mask_ = None
        else:
            done = False
            state_ = self.get_state(src_id=self.src_id, wk_id=self.wk_id, abs_doa=self.abs_doa)
            mask_ = self.get_mask(wk_id=self.wk_id, abs_doa=self.abs_doa)
        info = None
        
        return (state_, mask_), reward, done, info
    
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
