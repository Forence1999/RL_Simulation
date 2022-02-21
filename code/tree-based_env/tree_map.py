# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: d3qn.py
# @File: temp.py
# @Time: 2022/02/20/15:29
# @Software: PyCharm

import os, sys

F_dir = os.path.dirname(os.path.abspath(__file__))
FF_dir = os.path.dirname(F_dir)
sys.path.append(FF_dir)

import time
import random
import numpy as np
from copy import deepcopy
import networkx as nx
import pickle
from lib import utils
from matplotlib import pyplot as plt

EPS = np.finfo(float).eps * 4
global_ID = 20  # leave the previous IDs for Rooms
MIN_LINE_LENGTH = 40


# 新的基于树的环境与先前仅仅包含房间的环境有很大的不同：
# 在这个环境中，将每个房间看作是一个树的节点，并不断进行更加精细化地分为四叉树或者二叉树（基于房间长度是否大于预设的最短长度，以确保小车走过之后，该区域无人）。
# 对于每一个节点，都不知道自己的旁边是否有墙，旁边的节点又是谁，而只能向父节点查询，直到根节点。
# 每个节点的四个坐标均为顺时针方向：房间区域无起点保障，四叉树子节点从父节点中心开始，二叉树节点从父节点的中间线开始。
# 对于父节点来说，孩子的相对顺序并无保证，仅确定按照顺时针进行排列。


class Region(object):
    def __init__(self, vertex, center=None, parent=None, id=None):
        super(Region, self).__init__()
        
        self.id = self.__set_id__() if id is None else id
        # print(self.id)
        
        self.vertex = np.array(vertex)
        # print(self.vertex)
        self.line_centers = [Region.calculate_line_center(self.vertex[i], self.vertex[(i + 1) % 4]) for i in range(4)]
        self.center = Region.calculate_lines_cross_point([self.line_centers[0], self.line_centers[2]],
                                                         [self.line_centers[1], self.line_centers[3]],
                                                         exist_parallel=True, )  # 中心点并非四边形两条对角线的中点，而是两对对边中点相连的交点
        self.parent = parent
        self.children = self.__generate_children__()
        self.isLeaf = True if self.children is None else False
        self.num_children = 0 if self.isLeaf else len(self.children)
        self.center = self.center if center is None else center  # 生成子节点后更正center，以便于后续数据处理。因此，center不一定再是真正的子节点划分中心
    
    def __generate_children__(self, ):
        line1, line2 = self.vertex[:2], self.vertex[1:3]
        length1, length2 = Region.calculate_line_length(*line1, ), Region.calculate_line_length(*line2, )
        
        if length1 > MIN_LINE_LENGTH and length2 > MIN_LINE_LENGTH:
            vertex0 = [self.center, self.line_centers[3], self.vertex[0], self.line_centers[0]]
            vertex1 = [self.center, self.line_centers[0], self.vertex[1], self.line_centers[1]]
            vertex2 = [self.center, self.line_centers[1], self.vertex[2], self.line_centers[2]]
            vertex3 = [self.center, self.line_centers[2], self.vertex[3], self.line_centers[3]]
            
            return [Region(vertex0, parent=self), Region(vertex1, parent=self),
                    Region(vertex2, parent=self), Region(vertex3, parent=self), ]
        
        elif length1 > MIN_LINE_LENGTH and length2 <= MIN_LINE_LENGTH:
            vertex0 = [self.line_centers[0], self.line_centers[2], self.vertex[3], self.vertex[0], ]
            vertex1 = [self.line_centers[2], self.line_centers[0], self.vertex[1], self.vertex[2], ]
            
            return [Region(vertex0, parent=self), Region(vertex1, parent=self), ]
        
        elif length1 <= MIN_LINE_LENGTH and length2 > MIN_LINE_LENGTH:
            vertex0 = [self.line_centers[1], self.line_centers[3], self.vertex[0], self.vertex[1], ]
            vertex1 = [self.line_centers[3], self.line_centers[1], self.vertex[2], self.vertex[3], ]
            
            return [Region(vertex0, parent=self), Region(vertex1, parent=self), ]
        
        else:
            return None
    
    @staticmethod
    def calculate_line_length(point1, point2, ):
        '''
        :param point1: (x1, y1)
        :param point2: (x2, y2)
        :return: length of the line segment
        '''
        point1, point2, = np.asarray(point1), np.asarray(point2),
        return np.linalg.norm(point1 - point2, ord=2, axis=-1)
    
    @staticmethod
    def calculate_line_center(point1, point2, ):
        return np.mean([point1, point2], axis=0, )
    
    @staticmethod
    def calculate_general_line_form(point1, point2, ):
        (x1, y1), (x2, y2) = point1, point2,
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c
    
    @staticmethod
    def calculate_lines_cross_point(line1, line2, exist_parallel=True):
        '''
        :param line1: [(x1, y1), (x2, y2)]
        :param line2:
        :param exist_parallel: False: Force to return the intersection point at infinity
        :return:
        '''
        a1, b1, c1 = Region.calculate_general_line_form(*line1)
        a2, b2, c2 = Region.calculate_general_line_form(*line2)
        D = a1 * b2 - a2 * b1
        if abs(D) < EPS:
            if exist_parallel:
                print('Warning: The two lines are parallel')
                return None
            else:
                D = np.sign(D) * (abs(D) + EPS)
        
        x = (b1 * c2 - b2 * c1) / D
        y = (a2 * c1 - a1 * c2) / D
        return (x, y)
    
    def __set_id__(self, id=None, ):
        if id is not None:
            return id
        else:
            global global_ID
            global_ID += 1
            return global_ID
    
    def get_neighbor(self, ):
        return self.neighbors
    
    def get_center(self, ):
        return self.center
    
    def show(self, ):
        def DFS(node, region_info):
            '''
            Depth_First_Search to get all the region_info (id, vertexes, center)
            '''
            region_info.append((node.id, node.vertex, node.center,))
            
            if not node.isLeaf:
                for child in node.children:
                    DFS(child, region_info)
        
        region_info = []
        DFS(self, region_info)
        
        for id, vertex, center in region_info:
            color = [random.uniform(0, 1, ) for _ in range(3)]
            x, y = list(zip(*vertex))
            plt.fill(x, y, color=color, alpha=0.1)
            plt.text(*center, str(id), fontsize=10, ha='center', va='center', )
        plt.show()


class House(object):
    def __init__(self, ):
        super(House, self).__init__()
        self.id = 0
        self.center = (270, 225)
        centers = [None,  # 0
                   [60, 425],  # 1
                   [160, 320],  # 2
                   [340, 425],  # 3
                   [530, 320],  # 4
                   None,  # 5 [215, 220]
                   None,  # 6 [170, 160]
                   None,  # 7 [220, 100]
                   [280, 160],  # 8
                   [220, 15],  # 9
                   [460, 15],  # 10
                   None,  # 11 [420, 220]
                   [160, 425],  # 12
                   [530, 425],  # 13
                   [280, 220],  # 14
                   None,  # 15 [280, 100]
                   [280, 15],  # 16
                   [160, 220],  # 17
                   [530, 220],  # 18
                   None,  # 19 [170, 100]
                   [550, 15],  # 20
                   ]
        vertexes = [None,  # 0
                    [(4, 450), (145, 450), (145, 418), (4, 418)],  # 1
                    [(145, 418), (175, 418), (175, 250), (145, 250)],  # 2
                    [(175, 450), (515, 450), (515, 418), (175, 418)],  # 3
                    [(515, 418), (545, 418), (545, 250), (515, 250)],  # 4
                    None,  # 5
                    None,  # 6
                    None,  # 7
                    [(250, 210), (317, 210), (317, 42), (250, 42)],  # 8
                    [(180, 42), (250, 42), (250, 12), (180, 12)],  # 9
                    [(317, 42), (500, 42), (500, 12), (317, 12)],  # 10
                    None,  # 11
                    [(145, 450), (175, 450), (175, 418), (145, 418)],  # 12
                    [(515, 450), (545, 450), (545, 418), (515, 418)],  # 13
                    [(250, 250), (317, 250), (317, 210), (250, 210)],  # 14
                    None,  # 15
                    [(250, 42), (317, 42), (317, 12), (250, 12)],  # 16
                    [(145, 250), (175, 250), (175, 210), (145, 210)],  # 17
                    [(515, 250), (545, 250), (545, 210), (515, 210)],  # 18
                    None,  # 19
                    [(500, 42), (700, 42), (700, 12), (500, 12)],  # 20
                    ]
        self.centers = np.array(centers, dtype=object)
        self.vertexes = np.array(vertexes, dtype=object)
        self.num_room = len(self.centers) - sum(self.centers == None)
        self.rooms = self.__create_rooms__(vertexes=self.vertexes, centers=self.centers, parent=self)
        
        adjacency = np.load('./adjacency_list.npz')['adjacency_list']
        self.adjacency = np.array(adjacency)  # num_room * num_room: [i, j]: the direction (map_adj[i, j]) of i is j
        self.print_room_info()
        
        self.map = self.construct_map()  # a graph of map nodes to represent direct distance
    
    def __create_rooms__(self, vertexes, centers=None, parent=None):
        rooms = []
        centers = [None, ] * len(vertexes) if centers is None else centers
        for idx, (vertex, center) in enumerate(zip(vertexes, centers)):
            if vertex is None:
                rooms.append(None)
            else:
                rooms.append(Region(vertex=vertex, center=center, parent=parent, id=idx))
        
        return rooms
    
    def print_room_info(self, ):
        '''
        print the info of every room
        :return:
        '''
        print('-' * 20, 'Info of Rooms', '-' * 20, )
        for i, room in enumerate(self.rooms):
            if room is None:
                continue
            adjacency = self.adjacency[i]
            neighbors = np.full(shape=(8,), fill_value='', dtype=object)
            rooms = np.where(adjacency != np.inf)[0]
            directions = adjacency[rooms].astype(int)
            neighbors[directions] = list(rooms)
            print('id: ', room.id, '\n',
                  'directions: ', list(range(8)), '\n',
                  'neighbors:  ', neighbors)
        print('-' * 20, 'Finish printing graph', '-' * 20, )
    
    def construct_map(self, ):
        '''
        based on the adjacency and coordinates of nodes in the map, construct a graph with networkx and return it.
        :return: a networkx graph
        '''
        adjacency = np.array(self.adjacency)
        row, col = np.where(adjacency != np.inf)
        
        G = nx.Graph()
        nodes = {*row, *col}
        G.add_nodes_from(nodes)  # 节点索引 (id) 集合
        edges = [(i, j, Region.calculate_line_length(self.centers[i], self.centers[j])) for i, j in zip(row, col)]
        G.add_weighted_edges_from(edges)
        
        return G
    
    def show_map(self, ):
        pos_dict = dict(enumerate(self.centers))
        
        fig, ax = plt.subplots()
        nx.draw(self.map, ax=ax, with_labels=True, pos=pos_dict)
        plt.show()
    
    def show_rooms(self, ):
        def DFS(node, region_info):
            '''
            Depth_First_Search to get all the region_info (id, vertexes, center)
            '''
            region_info.append((node.id, node.vertex, node.center,))
            
            if not node.isLeaf:
                for child in node.children:
                    DFS(child, region_info)
        
        region_info = []
        for room in self.rooms:
            if room is None:
                continue
            DFS(room, region_info)
        
        plt.figure(dpi=300)
        for id, vertex, center in region_info:
            color = [random.uniform(0, 1, ) for _ in range(3)]
            x, y = list(zip(*vertex))
            plt.fill(x, y, color=color, alpha=0.1)
            plt.text(*center, str(id), fontsize=3, ha='center', va='center', )
        plt.show()
    
    def construct_tree(self, ):
        '''
        based on the adjacency and coordinates of nodes in the map, construct a graph with networkx and return it.
        :return: a networkx graph
        '''
        from treelib import Tree
        def DFS(node, region_info):
            '''
            Depth_First_Search to get all the region_info (id, vertexes, center)
            '''
            region_info.append((node.parent.id, node.parent.center, node.id, node.center,))
            
            if not node.isLeaf:
                for child in node.children:
                    DFS(child, region_info)
        
        region_info = []
        for room in self.rooms:
            if room is None:
                continue
            DFS(room, region_info)
        region_info = np.array(region_info, dtype=object)
        
        T = Tree(identifier=0)
        T.create_node(identifier=0, )
        for parent, _, child, _ in region_info:
            T.create_node(identifier=child, parent=parent, )
        
        # G = nx.DiGraph()
        # nodes = {*region_info[:, 0], *region_info[:, 2]}
        # G.add_nodes_from(nodes)  # 节点索引 (id) 集合
        # edges = [(i, j, 10) for i, ic, j, jc in region_info]
        # G.add_weighted_edges_from(edges)
        #
        # return G
        
        return T
    
    def show_tree(self, ):
        tree = self.construct_tree()
        # fig, ax = plt.subplots()
        # nx.draw(tree, ax=ax, with_labels=True, )
        # plt.show()
        tree.show()
    
    def show(self, ):
        self.show_map()
        # self.show_rooms()
        self.show_tree()


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
        centers = [[None, None],  # 0
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
        self.num_node = len(centers)
        self.coordinates = np.array(centers)
        map_adj = np.load('./adjacency_list.npz')['adjacency_list']
        self.map_adj = np.array(map_adj)  # [nodes, nodes]: store the distance between each pair of map nodes
        
        self.nodes = np.array([Region() for _ in range(self.num_node)])
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
    # import matplotlib.pyplot as plt
    #
    # ds_path = '../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
    # map_graph = Map_graph(ds_path=ds_path)
    # pos_dict = dict(enumerate(map_graph.coordinates))
    #
    # fig, ax = plt.subplots()
    # nx.draw(map_graph.map_graph, ax=ax, with_labels=True, pos=pos_dict)
    # plt.show()
    
    # vertex = [(0, 0), (0, 100), (100, 100), (100, 0)]
    # node = Region(vertex=vertex)
    # node.show()
    
    root = House()
    root.show()
    
    print('Brand-new World!')
