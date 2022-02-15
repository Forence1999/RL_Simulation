import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lib.utils import plot_curve
import tensorflow as tf

if __name__ == '__main__':
    
    base_model_dir = '/home/swang/project/SmartWalker/RL_Simulation_D3QN/model/base_model_fullData_woBN/classifier/ckpt'
    d3qn_model_dir = '/home/swang/project/SmartWalker/RL_Simulation_D3QN/model/d3qn_model/Dueling_DDQN_softUpdate__lr_0.0001_20220212-235802/classifier/ckpt'
    base_model = tf.keras.models.load_model(base_model_dir)
    d3qn_model = tf.keras.models.load_model(d3qn_model_dir)
    
    base_weights = base_model.get_weights()
    d3qn_weights = d3qn_model.get_weights()
    
    for i, j in zip(base_weights, d3qn_weights):
        if not np.allclose(i, j):
            print('not identical')
    
    pass
