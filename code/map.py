# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: RL_Simulation
# @File: Env_MAP.py
# @Time: 2021/10/31/10:19
# @Software: PyCharm


import os
import sys
import time
import random
import numpy as np
from copy import deepcopy
import networkx as nx
import pickle
from lib import utils


class Node(object):
    def __init__(self, ):
        super(Node, self).__init__()
    
    def set_coordinate(self, coordinate, ):
        self.coordinate = np.array(coordinate)
    
    def set_neighbors(self, neighbors, ):
        self.neighbors = np.array(neighbors)
    
    def set_id(self, id, ):
        self.id = id
    
    def get_neighbor(self, ):
        return self.neighbors
    
    def get_coordinate(self, ):
        return self.coordinate


class Map_graph(object):
    '''
    Map for 4F_CYC
    '''
    
    def __init__(self, ds_path, ):
        '''
        
        :param ds_path:
        '''
        super(Map_graph, self).__init__()
        self.ds_path = ds_path
        coordinates = [[None, None],  # 0
                       [60, 425],  # 1
                       [160, 320],  # 2
                       [340, 425],  # 3
                       [530, 320],  # 4
                       # [215, 220],  # 5
                       [None, None],  # 5
                       [170, 160],  # 6
                       [220, 100],  # 7
                       [280, 160],  # 8
                       [220, 15],  # 9
                       [460, 15],  # 10
                       # [420, 220],  # 11
                       [None, None],  # 11
                       [160, 425],  # 12
                       [530, 425],  # 13
                       [280, 220],  # 14
                       [280, 100],  # 15
                       [280, 15],  # 16
                       [160, 220],  # 17
                       [530, 220],  # 18
                       [170, 100],  # 19
                       [550, 15],  # 20
                       ]
        self.num_node = len(coordinates)
        self.coordinates = np.array(coordinates)
        map_adj = np.load('./adjacency_list.npz')['adjacency_list']
        self.map_adj = np.array(map_adj)  # [nodes, nodes]: store the distance between each pair of map nodes
        
        self.nodes = np.array([Node() for _ in range(self.num_node)])
        self.__init_nodes__()  # initialize every node in the graph (set_id, set_coordinate, set_neighbors)
        self.print_map_nodes()
        
        self.ds_info = self.get_src_wk_info()  # store src and wk info
        self.map_graph = self.construct_map_graph()  # a graph of map nodes to represent direct distance
        self.data_adj_ls = self.cal_data_adjacency_list()  # [src_nodes, wk_nodes]: store if wk_data exists for a src
        self.data_graph = self.construct_data_graph()  # a graph of map nodes to represent if data exists
        self.src_ids = np.where(np.any(self.data_adj_ls, axis=-1))[0]
        print('Number of src_ids: ', len(self.src_ids), '\n', 'src_ids: ', self.src_ids)
    
    def __init_nodes__(self, ):
        '''
        initialize every node in the graph (set_id, set_coordinate, set_neighbors, )
        :return:
        '''
        node_idx = np.arange(self.num_node)
        for i, adj_ls in enumerate(self.map_adj):
            if np.all(adj_ls == np.inf):
                self.nodes[i] = None
            else:
                neighbors = np.full((8,), None)
                neighbors[np.array(adj_ls[adj_ls != np.inf], dtype=int)] = node_idx[adj_ls != np.inf]
                # neighbors[np.array(adj_ls[adj_ls != np.inf], dtype=int)] = self.nodes[adj_ls != np.inf]
                self.nodes[i].set_neighbors(neighbors)
                self.nodes[i].set_coordinate(self.coordinates[i])
                self.nodes[i].set_id(i)
    
    def get_src_wk_info(self, ):
        '''
        store src and wk info to a dict
        :return:
        '''
        # get src and walker info
        ds_info = dict()
        src_paths = utils.get_subdirs_by_prefix(root=self.ds_path, prefix='src_')
        wk_paths = [utils.get_subdirs_by_prefix(root=i, prefix='walker_') for i in src_paths]
        for src, wk in zip(src_paths, wk_paths):
            ds_info[os.path.basename(src)] = [os.path.basename(i) for i in wk]
        
        return ds_info
    
    def construct_map_graph(self, ):
        '''
        based on the adjacency and coordinates of nodes in the map, construct a graph with networkx and return it.
        :return: a networkx graph
        '''
        map_adj = np.array(self.map_adj)
        row, col = np.array(np.where(map_adj != np.inf))
        
        G = nx.Graph()
        for i in {*row, *col}:  # 节点索引 (id) 集合
            G.add_node(i)
        for i, j in zip(row, col):
            distance = np.linalg.norm(self.coordinates[i] - self.coordinates[j], ord=2)
            G.add_weighted_edges_from([(i, j, distance)])
        
        return G
    
    def cal_data_adjacency_list(self, ):
        '''
        generate a 2D array ([src_nodes, wk_nodes]) to represent if wk_data exists for a src.
        :return:
        '''
        # cal_data_adjacency_list
        data_adj_ls = np.full((self.num_node, self.num_node), False)
        for src_key in self.ds_info.keys():
            src_coord = list(map(int, src_key.split('_')[1:3]))
            src_idx = np.where(np.all(self.coordinates == [src_coord], axis=-1))[0][0]
            for wk_key in self.ds_info[src_key]:
                wk_coord = list(map(int, wk_key.split('_')[1:3]))
                wk_coord = [wk_coord[0], wk_coord[-1]]
                wk_idx = np.where(np.all(self.coordinates == [wk_coord], axis=-1))[0][0]
                data_adj_ls[src_idx][wk_idx] = True
                # path = self.find_shortest_map_path(src_id=src_idx, wk_id=wk_idx)
                # for i in path:  # TODO: put every node in the path to data_adj_ls???
                #     self.data_adj_ls[src_idx][i] = True
        data_adj_ls[np.diag_indices_from(data_adj_ls)] = False
        
        print('-' * 20, 'Data graph', '-' * 20, )
        for src_id, adj_ls in enumerate(data_adj_ls):
            wk_ids = np.where(adj_ls)[0]
            if len(wk_ids) > 0:
                print('src:', src_id, '-' * 4, 'wk:', np.where(adj_ls)[0])
        print('-' * 20, 'Finish printing graph', '-' * 20, )
        
        return data_adj_ls
    
    def construct_data_graph(self, ):  # src -> wk
        '''
        construct a graph with networkx to represent the distance from src_node to walker_node
        :return:
        '''
        row, col = np.asarray(np.where(self.data_adj_ls))
        
        G = nx.DiGraph()
        for i in {*row, *col}:
            G.add_node(i)
        for i, j in list(zip(row, col)):
            distance = nx.dijkstra_path_length(self.map_graph, source=i, target=j)
            G.add_weighted_edges_from([(i, j, distance)])
        
        return G
    
    def print_map_nodes(self, ):
        '''
        print the info of every node
        :return:
        '''
        print('-' * 20, 'Graph of nodes', '-' * 20, )
        for i, node in enumerate(self.nodes):
            if node is None:
                continue
            print('id: ', node.id)
            print('directions: ', list(range(8)))
            # ids = []
            # for j, neighbor in enumerate(node.neighbors):
            #     if neighbor is not None:
            #         ids.append(neighbor.id)
            #     else:
            #         ids.append('')
            # print('neighbors: ', ids)
            print('neighbors:  ', node.neighbors)
        print('-' * 20, 'Finish printing graph', '-' * 20, )
    
    def find_shortest_map_path(self, src_id, wk_id, ):
        '''
        return the shortest path from the src_id to the wk_id.
        And the first and last nodes are src_id and wk_id respectively.
        :param src_id:
        :param wk_id:
        :return:
        '''
        path = nx.dijkstra_path(self.map_graph, source=src_id, target=wk_id)
        # distance = nx.dijkstra_path_length(self.map_graph, source=src_id, target=wk_id)
        return path
    
    def find_shortest_data_path(self, src_id, wk_id, ):
        '''
        return the shortest data path from the src_id to the wk_id
        :param src_id:
        :param wk_id:
        :return:
        '''
        path = nx.dijkstra_path(self.data_graph, source=src_id, target=wk_id)
        # distance = nx.dijkstra_path_length(self.data_graph, source=src_id, target=wk_id)
        return path
    
    def find_id_by_direction(self, base_id, direction, ):
        '''
        return id in the direction of the base_id
        :param base_id:
        :param direction:
        :return:
        '''
        return self.nodes[base_id].get_neighbor()[direction]
    
    def find_relative_direction(self, src_id, wk_id, ):
        '''
        find the direction of src_id relative to wk_id
        :param src_id:
        :param wk_id:
        :return:
        '''
        path = self.find_shortest_map_path(src_id, wk_id, )
        wk_neighbors = self.nodes[wk_id].get_neighbor()
        return np.where(wk_neighbors == path[-2])[0][0]
    
    def is_data_neighbor(self, src_id, wk_id, ):
        '''
        return if the data from src_id to wk_id exists. (cannot be guaranteed!)
        :param src_id:
        :param wk_id:
        :return:
        '''
        return self.data_adj_ls[src_id][wk_id]
    
    def find_intermediary_src(self, src_id, wk_id, ):
        '''
        return the intermediary nodes from src_id to wk_id
        :param src_id:
        :param wk_id:
        :return:
        '''
        path = self.find_shortest_data_path(src_id, wk_id)
        return path[1:-1]
    
    def get_coordinate(self, id, ):
        '''
        get the coordinates of a node.
        :param id:
        :return:
        '''
        return self.nodes[id].get_coordinate()
    
    def random_id(self, src_id=None, ):
        '''
        if src_id is None: return an id randomly.
        else: return one walker_id from src_id's data_children randomly
        :param src_id:
        :return:
        '''
        if src_id is None:
            return np.random.choice(self.src_ids, 1)[0]
        else:
            children = np.where(self.data_adj_ls[src_id], )[0]
            return np.random.choice(children, 1)[0]
    
    def random_doa(self, ):
        '''
        return a doa randomly.
        :return:
        '''
        return np.random.choice(np.arange(8), 1)[0]


if __name__ == '__main__':
    # adjacency_list = np.full((21, 21), np.inf)
    # for i in range(21):
    #     for j in range(21):
    #         ipt = input('{:.1f}相对于{:.1f}的位置关系：'.format(j, i))
    #         try:
    #             ipt = int(ipt)
    #             print(ipt)
    #             adjacency_list[i][j] = ipt
    #         except:
    #             pass
    # np.savez('./adjacency_list.npz', adjacency_list=adjacency_list)
    # print(adjacency_list)
    
    print('Hello World!')
    
    import matplotlib.pyplot as plt
    
    ds_path = '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
    map_graph = Map_graph(ds_path=ds_path)
    pos_dict = dict(enumerate(map_graph.coordinates))
    
    fig, ax = plt.subplots()
    nx.draw(map_graph.map_graph, ax=ax, with_labels=True, pos=pos_dict)
    plt.show()
    
    print('Brand-new World!')
