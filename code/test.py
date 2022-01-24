import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lib.utils import plot_curve

if __name__ == '__main__':
    
    path = './adjacency_list.npz'
    data = np.load(path)['adjacency_list']
    
    pass
