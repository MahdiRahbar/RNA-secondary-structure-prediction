# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0


import sys 
import os 
import numpy as np 
import pandas as pd
import torch 
from egnn_pytorch import EGNN_Network


net = EGNN_Network(
    num_tokens = 21,
    num_positions = 1024,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8,
    coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
)

def create_egnn_net(num_tokens, num_positions, dim, depth,
                     num_nearest_neighbors, coor_weights_clamp_value):
    net = EGNN_Network(
    num_tokens = 21,
    num_positions = 1024,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8,
    coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
    )
    return net

# feats = torch.randint(0, 21, (1, 1024)) # (1, 1024)
# coors = torch.randn(1, 1024, 3)         # (1, 1024, 3)
# mask = torch.ones_like(feats).bool()    # (1, 1024)

# feats_out, coors_out = net(feats, coors, mask = mask) # (1, 1024, 32), (1, 1024, 3)