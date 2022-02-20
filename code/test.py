import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lib.utils import plot_curve
import tensorflow as tf

if __name__ == '__main__':
    data = np.load('/home/swang/project/SmartWalker/RL_Simulation/code/adjacency_list.npz')['adjacency_list']
    
    np.savez('/home/swang/project/SmartWalker/RL_Simulation/code/adjacency_list.npz', adjacency_list=data, )
    pass
