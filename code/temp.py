import os, sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy


# 新的基于树的环境与先前仅仅包含房间的环境有很大的不同：
# 在这个环境中，将每个房间看作是一个树的节点，并不断进行更加精细化地分为四叉树或者二叉树（基于房间长度是否大于预设的最短长度，以确保小车走过之后，该区域无人）。
# 对于每一个节点，都不知道自己的旁边是否有墙，旁边的节点又是谁，而只能向父节点查询，直到根节点。
# 每个节点的四个坐标均为顺时针方向：房间区域无起点保障，四叉树子节点从父节点中心开始，二叉树节点从父节点的中间线开始。
# 对于父节点来说，孩子的相对顺序并无保证，仅确定按照顺时针进行排列。


def test(ls):
    a = ls
    a.append(1)


if __name__ == '__main__':
    
    print('Hello World!')
    ls = []
    for i in range(5):
        test(ls)
    print(ls)
    
    print('Brand-new World!')
