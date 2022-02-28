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
MAX_LINE_LENGTH = 40  # maximum length of the side of one area


class Area(object):
    def __init__(self, vertex, center=None, parent=None, id=None):
        super().__init__()
        
        self.id = self.__set_id__() if id is None else id
        # print(self.id)
        
        self.vertex = np.array(vertex)
        # print(self.vertex)
        self.line_centers = [Area.calculate_line_center(self.vertex[i], self.vertex[(i + 1) % 4]) for i in range(4)]
        self.center = Area.calculate_lines_cross_point([self.line_centers[0], self.line_centers[2]],
                                                       [self.line_centers[1], self.line_centers[3]],
                                                       exist_parallel=True, )
        # This center point is not the midpoint of the two diagonals of the quadrilateral,
        # but the intersection of the midpoints of the two pairs of opposite sides.
        self.parent = parent
        self.neighbors = None
        self.children = self.__generate_children__()
        self.isLeaf = True if self.children is None else False
        self.num_children = 0 if self.isLeaf else len(self.children)
        self.center = self.center if center is None else center  # This adjusted center is not the one for child division any more.
    
    def __generate_children__(self, ):
        line1, line2, line3, line4 = self.vertex[:2], self.vertex[1:3], self.vertex[2:4], self.vertex[[3, 0]]
        length1, length2 = max(Area.calculate_line_length(*line1, ), Area.calculate_line_length(*line3, )), \
                           max(Area.calculate_line_length(*line2, ), Area.calculate_line_length(*line4, ))
        
        if length1 > MAX_LINE_LENGTH and length2 > MAX_LINE_LENGTH:
            vertex0 = [self.center, self.line_centers[3], self.vertex[0], self.line_centers[0]]
            vertex1 = [self.center, self.line_centers[0], self.vertex[1], self.line_centers[1]]
            vertex2 = [self.center, self.line_centers[1], self.vertex[2], self.line_centers[2]]
            vertex3 = [self.center, self.line_centers[2], self.vertex[3], self.line_centers[3]]
            
            return [Area(vertex0, parent=self), Area(vertex1, parent=self),
                    Area(vertex2, parent=self), Area(vertex3, parent=self), ]
        
        elif length1 > MAX_LINE_LENGTH and length2 <= MAX_LINE_LENGTH:
            vertex0 = [self.line_centers[0], self.line_centers[2], self.vertex[3], self.vertex[0], ]
            vertex1 = [self.line_centers[2], self.line_centers[0], self.vertex[1], self.vertex[2], ]
            
            return [Area(vertex0, parent=self), Area(vertex1, parent=self), ]
        
        elif length1 <= MAX_LINE_LENGTH and length2 > MAX_LINE_LENGTH:
            vertex0 = [self.line_centers[1], self.line_centers[3], self.vertex[0], self.vertex[1], ]
            vertex1 = [self.line_centers[3], self.line_centers[1], self.vertex[2], self.vertex[3], ]
            
            return [Area(vertex0, parent=self), Area(vertex1, parent=self), ]
        
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
        a1, b1, c1 = Area.calculate_general_line_form(*line1)
        a2, b2, c2 = Area.calculate_general_line_form(*line2)
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
        def DFS(node, area_info):
            '''
            Depth_First_Search to get all the area_info (id, vertexes, center)
            '''
            area_info.append((node.id, node.vertex, node.center,))
            
            if not node.isLeaf:
                for child in node.children:
                    DFS(child, area_info)
        
        area_info = []
        DFS(self, area_info)
        
        for id, vertex, center in area_info:
            color = [random.uniform(0, 1, ) for _ in range(3)]
            x, y = list(zip(*vertex))
            plt.fill(x, y, color=color, alpha=0.1)
            plt.text(*center, str(id), fontsize=10, ha='center', va='center', )
        plt.show()


class House(object):
    def __init__(self, centers=None, vertexes=None, ):
        super().__init__()
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
                   ] if centers is None else centers
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
                    ] if vertexes is None else vertexes
        self.centers = np.array(centers, dtype=object)
        self.vertexes = np.array(vertexes, dtype=object)
        self.num_room = len(self.centers) - sum(self.centers == None)
        self.rooms = self.__create_rooms__(vertexes=self.vertexes, centers=self.centers, parent=self)
        
        adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './adjacency_list.npz')
        adjacency = np.load(adj_path)['adjacency_list']
        self.adjacency = np.array(adjacency)  # num_room * num_room: [i, j]: the direction (map_adj[i, j]) of i is j
        # self.print_house_info()
        
        self.graph = self.construct_graph()  # a graph of map nodes to represent direct distance
        self.set_neighbors()
        self.leaf_graph = self.construct_leaf_graph()
        self.areas = self.collect_areas()
    
    def __create_rooms__(self, vertexes, centers=None, parent=None):
        rooms = []
        centers = [None, ] * len(vertexes) if centers is None else centers
        for idx, (vertex, center) in enumerate(zip(vertexes, centers)):
            if vertex is None:
                rooms.append(None)
            else:
                rooms.append(Area(vertex=vertex, center=center, parent=parent, id=idx))
        
        return np.array(rooms, dtype=object)
    
    def get_nodes(self):
        def condition(**kwargs):
            return True
        
        node_ls = []
        for room in self.rooms:
            if room is not None:
                self.DFS(room, node_ls, condition)
        
        return node_ls
    
    def DFS(self, root, node_ls, condition: callable = None):
        ''' Depth_First_Search to get all the leaves '''
        
        if condition is None or condition(node=root):
            node_ls.append(root)
        if not root.isLeaf:
            for child in root.children:
                self.DFS(child, node_ls, condition)
    
    def get_leaves(self, ):
        leaf_ls = []
        for room in self.rooms:
            if room is not None:
                leaf_ls.extend(self.get_room_leaves(room=room))
        return leaf_ls
    
    def get_room_leaves(self, room):
        condition = lambda node: node.isLeaf
        
        leaf_ls = []
        self.DFS(room, leaf_ls, condition)
        
        return leaf_ls
    
    def print_house_info(self, ):
        '''
        print the info of every room
        :return:
        '''
        print('-' * 20, 'Info of House', '-' * 20, )
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
        print('-' * 20, 'Finish printing info of House', '-' * 20, )
    
    def construct_graph(self, ):
        '''
        based on the adjacency and coordinates of nodes in the map, construct a graph with networkx and return it.
        :return: a networkx graph
        '''
        adjacency = np.array(self.adjacency)
        row, col = np.where(adjacency != np.inf)
        
        G = nx.Graph()
        nodes = {*row, *col}
        G.add_nodes_from(nodes)  # 节点索引 (id) 集合
        edges = [(i, j, Area.calculate_line_length(self.centers[i], self.centers[j])) for i, j in zip(row, col)]
        G.add_weighted_edges_from(edges)
        
        return G
    
    def construct_tree(self, ):
        '''
        based on the adjacency and coordinates of nodes in the map, construct a graph with networkx and return it.
        :return: a networkx graph
        '''
        from treelib import Tree
        def DFS(node, area_info):
            '''
            Depth_First_Search to get all the area_info (id, vertexes, center)
            '''
            area_info.append((node.parent.id, node.parent.center, node.id, node.center,))
            
            if not node.isLeaf:
                for child in node.children:
                    DFS(child, area_info)
        
        area_info = []
        for room in self.rooms:
            if room is None:
                continue
            DFS(room, area_info)
        area_info = np.array(area_info, dtype=object)
        
        T = Tree(identifier=0)
        T.create_node(identifier=0, )
        for parent, _, child, _ in area_info:
            T.create_node(identifier=child, parent=parent, )
        
        # G = nx.DiGraph()
        # nodes = {*area_info[:, 0], *area_info[:, 2]}
        # G.add_nodes_from(nodes)  # 节点索引 (id) 集合
        # edges = [(i, j, 10) for i, ic, j, jc in area_info]
        # G.add_weighted_edges_from(edges)
        #
        # return G
        
        return T
    
    def show_graph(self, ):
        pos_dict = dict(enumerate(self.centers))
        
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True, pos=pos_dict)
        plt.show()
    
    def show_leaf_graph(self, ):
        nodes = self.leaf_graph.nodes()
        pos_dict = dict([(i, np.array(self.areas[i].center) * 10) for i in nodes])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw(self.leaf_graph, ax=ax, with_labels=True, pos=pos_dict)
        plt.show()
    
    def show_rooms(self, ):
        nodes = self.get_nodes()
        area_info = [(node.id, node.vertex, node.center,) for node in nodes]
        
        plt.figure(dpi=300)
        for id, vertex, center in area_info:
            color = [random.uniform(0, 1, ) for _ in range(3)]
            x, y = list(zip(*vertex))
            plt.fill(x, y, color=color, alpha=0.1)
            plt.text(*center, str(id), fontsize=3, ha='center', va='center', )
        plt.show()
    
    def show_tree(self, ):
        tree = self.construct_tree()
        # fig, ax = plt.subplots()
        # nx.draw(tree, ax=ax, with_labels=True, )
        # plt.show()
        tree.show()
    
    def show_leaves(self, ):
        leaves = self.get_leaves()
        leaf_info = [(leaf.id, leaf.vertex, leaf.center,) for leaf in leaves]
        
        plt.figure(dpi=300)
        for id, vertex, center in leaf_info:
            color = [random.uniform(0, 1, ) for _ in range(3)]
            x, y = list(zip(*vertex))
            plt.fill(x, y, color=color, alpha=0.1)
            plt.text(*center, str(id), fontsize=3, ha='center', va='center', )
        plt.show()
    
    def show(self, ):
        # self.show_graph()
        # self.show_rooms()
        # self.show_tree()
        # self.show_leaves()
        # self.show_neighbors_of_leaves()
        self.show_leaf_graph()
    
    def set_neighbors(self, ):
        ''' For every leaf, find its neighbors and set them in the leaf. '''
        
        def find_node_of_direction(img, id, center, transform):
            def verify_PointLegitimacy(point, img):
                x, y = point
                return x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]
            
            def verify_EndSearch(counter: dict):
                for key in counter.keys():
                    if counter[key] > 10:
                        return key
                    elif (key == -1) and counter[key] > 3:
                        return key
                return False
            
            x, y = np.array(center)
            counter = dict()
            while verify_PointLegitimacy(point=(x, y), img=img):
                if img[x, y] == id:
                    counter = dict()
                    x, y = transform(x, y)
                    continue
                elif img[x, y] == -2:
                    x, y = transform(x, y)
                    continue
                elif img[x, y] == -1:
                    counter[-1] = counter[-1] + 1 if -1 in counter else 1
                else:
                    counter[-1] = 0
                    counter[img[x, y]] = counter[img[x, y]] + 1 if img[x, y] in counter else 1
                if verify_EndSearch(counter) is not False:
                    break
                x, y = transform(x, y)
            res = verify_EndSearch(counter)
            # return None if ((res is False) or (res == -1)) else res
            return None if ((res is False) or (res == -1)) else res
        
        # plot every leaf on the image
        import cv2
        
        leaves = self.get_leaves()
        leaf_info = [(leaf.id, leaf.vertex, leaf.center,) for leaf in leaves]
        ids, vertexes, centers = list(zip(*leaf_info))
        y_max, x_max = np.ceil(np.max(vertexes, axis=(0, 1))).astype(int)
        
        leaf_img = np.full(shape=(x_max, y_max,), fill_value=-1, dtype=np.int32)
        cv2.fillConvexPoly(leaf_img, np.array([(175, 250), (250, 250), (250, 210), (175, 210)]),
                           -2)  # for deprecated areas(5 & 11)
        cv2.fillConvexPoly(leaf_img, np.array([(317, 250), (515, 250), (515, 210), (317, 210)]), -2)
        for id, vertex, _ in leaf_info:
            vertex = np.rint(vertex).astype(int)
            cv2.fillConvexPoly(leaf_img, vertex, id)
        plt.imshow(leaf_img)
        plt.show()
        
        # print('set neighbors...')
        neighbors_ls = []
        for id, _, center in leaf_info:
            y, x = np.rint(center).astype(int)
            transforms = [lambda x, y: (x, y + 1),  # 0
                          lambda x, y: (x + 1, y + 1),  # 1
                          lambda x, y: (x + 1, y),  # 2
                          lambda x, y: (x + 1, y - 1),  # 3
                          lambda x, y: (x, y - 1),  # 4
                          lambda x, y: (x - 1, y - 1),  # 5
                          lambda x, y: (x - 1, y),  # 6
                          lambda x, y: (x - 1, y + 1),  # 7
                          ]
            neighbors = [find_node_of_direction(leaf_img, id, (x, y), transform) for transform in transforms]
            neighbors_ls.append(neighbors)
            # print(id, '\n', list(range(8)), '\n', neighbors)
        for leaf, neighbors in zip(leaves, neighbors_ls):
            neighbor_leafs = []
            for neighbor_id in neighbors:
                if neighbor_id is None:
                    neighbor_leafs.append(None)
                else:
                    neighbor_leaf = leaves[np.where(neighbor_id == ids)[0][0]]
                    neighbor_leafs.append(neighbor_leaf)
            leaf.neighbors = neighbor_leafs
        # print('set neighbors done.')
    
    def show_neighbors_of_leaves(self, ):
        leaves = self.get_leaves()
        leaves.sort(key=lambda leaf: leaf.id)
        for leaf in leaves:
            neighbor_ids = []
            for neighbor in leaf.neighbors:
                if neighbor is None:
                    neighbor_ids.append(None)
                else:
                    neighbor_ids.append(neighbor.id)
            print(leaf.id, '\n', list(range(8)), '\n', neighbor_ids)
    
    def find_room(self, id=None, area=None, ):
        '''
        area enjoys a higher priority than id.
        '''
        area = area if area is not None else self.get_area_by_id(id)
        assert isinstance(area, Area), f'Area (id: {id}) or area itself is not an instance of Area.'
        
        inter_area = area
        while inter_area.parent is not self:
            inter_area = inter_area.parent
        
        return inter_area
    
    def collect_areas(self, ):
        ''' organize all the areas in the tree into a list. And index them by their ids. '''
        nodes = self.get_nodes()
        
        id_max = max([node.id for node in nodes])
        area_ls = [None] * (id_max + 1)
        for node in nodes:
            area_ls[node.id] = node
        area_ls[self.id] = self
        
        return area_ls
    
    def get_area_by_id(self, id):
        return self.areas[id]
    
    def inSameRoom(self, id1, id2):
        room_id_1 = self.find_room(id=id1).id
        room_id_2 = self.find_room(id=id2).id
        
        return room_id_1 == room_id_2
    
    def random_area(self, ):
        '''
        return a random id in the map.
        :return:
        '''
        # There are two ways to get a random area:
        # 1. randomly choose one area from all the leaves;
        # 2. randomly choose one room first before choosing one leaf from its leaves.
        
        # room = np.random.choice(self.rooms, p=np.array(self.rooms != None, dtype=float))
        room = random.choices(self.rooms, weights=np.array(self.rooms != None, dtype=float), k=1)[0]
        leaves = self.get_room_leaves(room=room)
        leaf = random.choice(leaves)
        # print('Room_id:', room.id, 'Leaf:', leaf.id)
        
        return leaf
    
    def get_neighbors(self, id):
        ''' Get the neighbors of the area with id. '''
        area = self.get_area_by_id(id=id)
        return area.neighbors
    
    def construct_leaf_graph(self, ):
        leaves = self.get_leaves()
        
        G = nx.Graph()
        for leaf in leaves:
            for neighbor in leaf.neighbors:
                if neighbor is None:
                    continue
                G.add_edge(leaf.id, neighbor.id)
        
        return G


class Dataset_Graph(object):
    def __init__(self, ds_path, map_graph, centers=None):
        super().__init__()
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
                   ] if centers is None else centers
        self.centers = np.array(centers, dtype=object)
        self.num_node = len(centers)
        
        self.ds_path = ds_path
        self.ds_info = self.get_src_wk_info()  # store src and wk info
        self.data_adj_ls = self.cal_data_adjacency_list()  # [src_nodes, wk_nodes]: store if wk_data exists for a src
        self.graph = self.construct_data_graph(map_graph)  # a graph of map nodes to represent if data exists
        
        self.src_ids = np.where(np.any(self.data_adj_ls, axis=-1))[0]
        # print('Number of src_ids: ', len(self.src_ids), '\n', 'src_ids: ', self.src_ids)
        # self.print_data_info()
    
    def get_src_wk_info(self, ):
        ''' store src and wk info to a dict '''
        
        # get src and walker info
        ds_info = dict()
        src_paths = utils.get_subdirs_by_prefix(root=self.ds_path, prefix='src_')
        wk_paths = [utils.get_subdirs_by_prefix(root=i, prefix='walker_') for i in src_paths]
        for src, wk in zip(src_paths, wk_paths):
            ds_info[os.path.basename(src)] = [os.path.basename(i) for i in wk]
        
        return ds_info
    
    def cal_data_adjacency_list(self, ):
        '''
        generate a 2D array ([src_nodes, wk_nodes]) to represent if wk_data exists for a src.
        :return:
        '''
        # cal_data_adjacency_list
        
        center_index = np.where(self.centers != None, )[0]
        centers = np.array(list(self.centers[center_index]))
        data_adj_ls = np.full((self.num_node, self.num_node), False)
        for src_key in self.ds_info.keys():
            src_coord = list(map(int, src_key.split('_')[1:3]))
            src_idx = center_index[np.where(np.all(centers == [src_coord], axis=-1))[0][0]]
            for wk_key in self.ds_info[src_key]:
                wk_coord = list(map(int, wk_key.split('_')[1:3]))
                wk_idx = center_index[np.where(np.all(centers == [wk_coord], axis=-1))[0][0]]
                data_adj_ls[src_idx][wk_idx] = True
            data_adj_ls[np.diag_indices_from(data_adj_ls)] = False
        
        return data_adj_ls
    
    def print_data_info(self, ):
        ''' print the info of data '''
        print('-' * 20, 'Data graph', '-' * 20, )
        for src_id, adj_ls in enumerate(self.data_adj_ls):
            wk_ids = np.where(adj_ls)[0]
            if len(wk_ids) > 0:
                print('src:', src_id, '-' * 4, 'wk:', np.where(adj_ls)[0])
        print('-' * 20, 'Finish printing Data graph', '-' * 20, )
    
    def construct_data_graph(self, map_graph):  # src -> wk
        ''' construct a graph with networkx to represent the distance from src_node to walker_node '''
        row, col = np.asarray(np.where(self.data_adj_ls))
        
        G = nx.DiGraph()
        nodes = {*row, *col}
        G.add_nodes_from(nodes)  # 节点索引 (id) 集合
        edges = [(i, j, nx.dijkstra_path_length(map_graph, source=i, target=j)) for i, j in zip(row, col)]
        G.add_weighted_edges_from(edges)
        
        return G


class Map_graph(object):
    '''
    Map for 4F_CYC
    '''
    
    def __init__(self, ds_path, ):
        super().__init__()
        self.ds_path = ds_path
        self.centers = [None,  # 0
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
        self.vertexes = [None,  # 0
        
                         [(4, 450), (145, 450), (145, 418), (4, 418)],  # 1
                         [(145, 418), (175, 418), (175, 250), (145, 250)],  # 2
                         [(175, 450), (515, 450), (515, 418), (175, 418)],  # 3
                         [(515, 418), (545, 418), (545, 250), (515, 250)],  # 4
                         None,  # [(175, 250), (250, 250), (250, 210), (175, 210)],  # 5
                         None,  # 6
                         None,  # 7
                         [(250, 210), (317, 210), (317, 42), (250, 42)],  # 8
                         [(180, 42), (250, 42), (250, 12), (180, 12)],  # 9
                         [(317, 42), (500, 42), (500, 12), (317, 12)],  # 10
                         None,  # [(317, 250), (515, 250), (515, 210), (317, 210)],  # 11
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
        self.house = House(centers=self.centers, vertexes=self.vertexes)
        self.map_graph = self.house.graph
        self.dataset_graph = Dataset_Graph(ds_path=self.ds_path, map_graph=self.map_graph, centers=self.centers)
        self.data_graph = self.dataset_graph.graph
    
    def find_shortest_map_path(self, src_id, wk_id, ):
        '''
        return the shortest path from the src_id to the wk_id.
        And the first and last nodes are src_id and wk_id respectively.
        :param src_id:
        :param wk_id:
        :return:
        '''
        src_room_id, wk_room_id = self.find_room(id=src_id).id, self.find_room(id=wk_id).id
        path = nx.dijkstra_path(self.map_graph, source=src_room_id, target=wk_room_id)
        # distance = nx.dijkstra_path_length(self.map_graph, source=src_id, target=wk_id)
        return path
    
    def find_shortest_data_path(self, src_id, wk_id, ):
        ''' return the shortest data path from the src_id to the wk_id '''
        src_room_id, wk_room_id = self.find_room(id=src_id).id, self.find_room(id=wk_id).id
        
        path = nx.dijkstra_path(self.data_graph, source=src_room_id, target=wk_room_id)
        # distance = nx.dijkstra_path_length(self.data_graph, source=src_id, target=wk_id)
        return path
    
    def find_id_by_direction(self, base_id, direction, ):
        '''
        return id in the direction of the base_id
        :param base_id:
        :param direction:
        :return:
        '''
        neighbors = self.get_area_by_id(id=base_id).neighbors
        return neighbors[direction].id
    
    def find_relative_direction(self, src_id, wk_id, ):
        '''
        find the direction of src_id relative to wk_id
        :param src_id:
        :param wk_id:
        :return:
        '''
        src_room_id, wk_room_id = self.find_room(id=src_id).id, self.find_room(id=wk_id).id
        path = self.find_shortest_map_path(src_id=src_room_id, wk_id=wk_room_id, )
        
        return int(self.house.adjacency[wk_room_id, path[-2]])
    
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
    
    def get_center(self, id, ):
        ''' get the center coordinates of a area. '''
        return self.house.areas[id].center
    
    def get_room_center(self, id, ):
        ''' get the center coordinates of a room where the area of id is located. '''
        room = self.find_room(id=id)
        return self.get_center(id=room.id)
    
    def random_id(self, ):
        ''' return a random id in the map. And the selection method is defined in House class. '''
        return self.house.random_area().id
    
    def random_doa(self, ):
        ''' return a doa randomly. '''
        return random.choice(list(range(8)))
    
    def inSameRoom(self, id1, id2):
        return self.house.inSameRoom(id1, id2)
    
    def find_room(self, id=None, area=None, ):
        return self.house.find_room(id=id, area=area, )
    
    def get_area_by_id(self, id):
        return self.house.get_area_by_id(id=id)
    
    def get_neighbors(self, id):
        return self.house.get_neighbors(id=id)


if __name__ == '__main__':
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
    # node = Area(vertex=vertex)
    # node.show()
    
    ds_path = '../../dataset/4F_CYC/1s_0.5_800_16000/ini_hann_norm_denoise_drop_stft_seglen_64ms_stepsize_ratio_0.5'
    
    # root = House()
    # root.show()
    # data = Dataset_Graph(ds_path=ds_path, map_graph=root.graph, centers=root.centers)
    
    map_graph = Map_graph(ds_path=ds_path)
    map_graph.house.show()
    
    print('Brand-new World!')
