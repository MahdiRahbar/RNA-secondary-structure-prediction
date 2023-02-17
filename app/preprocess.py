# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0

"""
    This is an overall description the preprocessing file
"""

import os
import sys
import numpy as np
import datetime
import argparse
import random
import operator
import itertools
import operator
import pandas as pd
flag_plots = False



import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from time import sleep
from tqdm import tqdm

from tensorflow.keras.utils import *
import tensorflow as tf

from tensorflow.keras import backend as K


if flag_plots:
    #%matplotlib inline
    from plots import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)


#http://bioinformatics.hitsz.edu.cn/repRNA/static/download/physicochemical_property_indices.pdf
#	Adenine_content	 Cytosine content (3) 	Enthalpy (4) 	Enthalpy2 (4) 	Entropy (5) 	Entropy2 (5)	Free energy(5)	 Free energy2(5) 	GCcontent(3)	 Guaninecontent(3) Hy	drophilicity(2)	Hydrophilicity2	Keto (GT)content	Purine (AG)content	Rise (1)	Roll (1)	Shift (1)	Slide (1)	Stacking energy (1)	Thymine content (3)	Tilt (1)	Twist (1)

physicochemical_property_indices = {'AA':[2,0,-6.6,-6.82,-18.4,-19,-0.9,-0.93,0,0,0.023,0.04,0,2,3.18,7,-0.08,-1.27,-13.7,0,-0.8,31],
'AC':[1,1,-10.2,-11.4,-26.2,-29.5,-2.1,-2.24,1,0,0.083,0.14,0,1,3.24,4.8,0.23,-1.43,-13.8,0,0.8,32],
'AG':[1,0,-7.6,-10.48,-19.2,-27.1,-1.7,-2.08,1,1,0.035,0.08,0,2,3.3,8.5,-0.04,-1.5,-14,0,0.5,30],
'AU':[1,0,-5.7,-9.38,-15.5,-26.7,-0.9,-1.1,0,0,0.09,0.14,1,1,3.24,7.1,-0.06,-1.36,-15.4,1,1.1,33],
'CA':[1,1,-10.5,-10.44,-27.8,-26.9,-1.8,-2.11,1,0,0.118,0.21,0,1,3.09,9.9,0.11,-1.46,-14.4,0,1,31],
'CC':[0,2,-12.2,-13.39,-29.7,-32.7,-2.9,-3.26,2,0,0.349,0.49,0,0,3.32,8.7,-0.01,-1.78,-11.1,0,0.3,32],
'CG':[0,1,-8,-10.64,-19.4,-26.7,-2,-2.36,2,1,0.193,0.35,1,1,3.3,12.1,0.3,-1.89,-15.6,0,-0.1,27],
'CU':[0,1,-7.6,-10.48,-19.2,-27.1,-1.7,-2.08,1,0,0.378,0.52,1,0,3.3,8.5,-0.04,-1.5,-14,1,0.5,30],
'GA':[1,0,-13.3,-12.44,-35.5,-32.5,-2.3,-2.35,1,1,0.048,0.1,1,2,3.38,9.4,0.07,-1.7,-14.2,0,1.3,32],
'GC':[0,1,-14.2,-14.88,-34.9,-36.9,-3.4,-3.42,2,1,0.146,0.26,1,1,3.22,6.1,0.07,-1.39,-16.9,0,0,35],
'GG':[0,0,-12.2,-13.39,-29.7,-32.7,-2.9,-3.26,2,2,0.065,0.17,2,2,3.32,12.1,-0.01,-1.78,-11.1,0,0.3,32],
'GU':[0,0,-10.2,-11.4,-26.2,-29.5,-2.1,-2.24,1,1,0.16,0.27,2,1,3.24,4.8,0.23,-1.43,-13.8,1,0.8,32],
'UA':[1,0,-8.1,-7.69,-22.6,-20.5,-1.1,-1.33,0,0,0.112,0.21,1,1,3.26,10.7,-0.02,-1.45,-16,1,-0.2,32],
'UC':[0,1,-10.2,-12.44,-26.2,-32.5,-2.1,-2.35,1,0,0.359,0.48,1,0,3.38,9.4,0.07,-1.7,-14.2,1,1.3,32],
'UG':[0,0,-7.6,-10.44,-19.2,-26.9,-1.7,-2.11,1,1,0.224,0.34,1,1,3.09,9.9,0.11,-1.46,-14.4,1,1,31],
'UU':[0,0,-6.6,-6.82,-18.4,-19,-0.9,-0.93,0,0,0.389,0.44,2,0,3.18,7,-0.08,-1.27,-13.7,2,-0.8,31]}


import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
expected_n_channels =4 

from .config import *

def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='EXAMPLE:\npython3 /faculty/jhou4/Projects/RNA_folding/train_network_Nov28_SingleTrain.py  -b 5 -r 0-50 -n -1 -c 64 -e 10 -d 4 -f 8 -p /faculty/jhou4/Projects/RNA_folding/data/Own_data/Sharear -v 0 -o /faculty/jhou4/Projects/RNA_folding/train_results_20201128_len0_50_4')
    #parser.add_argument('-w', type=str, required = True, dest = 'file_weights', help="hdf5 weights file")
    parser.add_argument('-b', type=int, required = True, dest = 'batch_size', help="number of pdbs to use for each batch")
    parser.add_argument('-r', type=str, required = True, dest = 'len_range', help="lengths of pdbs to use for each batch")
    parser.add_argument('-n', type=int, required = True, dest = 'dev_size', help="number of pdbs to use for training (use -1 for ALL)")
    parser.add_argument('-c', type=int, required = True, dest = 'training_window', help="crop size (window) for training, 64, 128, etc. ")
    parser.add_argument('-e', type=int, required = True, dest = 'training_epochs', help="# of epochs")
    parser.add_argument('-o', type=str, required = True, dest = 'dir_out', help="directory to write .npy files")
    parser.add_argument('-d', type=int, required = True, dest = 'arch_depth', help="residual arch depth")
    parser.add_argument('-f', type=int, required = True, dest = 'filters_per_layer', help="number of convolutional filters in each layer")
    parser.add_argument('-p', type=str, required = True, dest = 'data_path', help="path where all the data (including .lst) is located")
    parser.add_argument('-v', type=int, required = True, dest = 'flag_eval_only', help="1 = Evaluate only, don't train")
    parser.add_argument('-a', type=int, required = True, dest = 'flag_noncanonical', help="1 = Train/Evaluate on canonical pairs")
    parser.add_argument('-z', type=int, required = True, dest = 'weight_regularization', help="weight regularization")
    parser.add_argument('-k', type=int, required = True, dest = 'filter_size_2d', help="filter_size_2d")
    parser.add_argument('-l', type=float, required = True, dest = 'loss_ratio', help="ratio in weighted loss")
    parser.add_argument('-t', type=float, required = True, dest = 'dropout_rate', help="ratio in dropout")
    parser.add_argument('-m', type=int, required = True, dest = 'lstm_layers', help="number of lstm layers")
    parser.add_argument('-y', type=int, required = True, dest = 'fully_layers', help="number of fully connected layers")
    parser.add_argument('-g', type=float, required = True, dest = 'nt_reg_weight', help="weight for nt regularized term")
    parser.add_argument('-u', type=float, required = True, dest = 'pair_reg_weight', help="weight for pair_reg_weight  regularized term")
    parser.add_argument('-j', type=int, required = True, dest = 'lstm_filter', help="filter size for lstm")
    parser.add_argument('-s', type=int, required = True, dest = 'feature_type', help="filter type")
    parser.add_argument('-i', type=int, required = True, dest = 'include_pseudoknots', help="filter type")
    parser.add_argument('-q', type=int, required = True, dest = 'dilation_size', help="dilation")
    
    args = parser.parse_args()
    return args






def weighted_binary_crossentropy_ntRegularized(y_true, y_pred) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    if not y_true.dtype == tf.float32:
        y_true = K.cast(y_true, tf.float32)
    if not y_pred.dtype == tf.float32:
        y_pred = K.cast(y_pred, tf.float32)
    logloss = -(1 - y_true) * K.log(1 - y_pred) 
    return K.mean( logloss, axis=-1)



def weighted_binary_crossentropy_pairRegularized(y_true, y_pred):
    #import tensorflow as tf

    #x = tf.constant([[-1, 2, -1, 4],[1, -1, 3, -1]])
    #y = tf.Variable([[0.1, 0.2, 0.1, 0.4],[0.1, 0.1, 0.3, 0.1]])
    
    #masked = tf.not_equal(y_pred,-1)
    #masked2 = tf.equal(y_pred,-1)
    masked = tf.not_equal(y_true,-1)
    masked2 = tf.equal(y_true,-1)
    zeros = tf.zeros_like(y_true)
    zeros = tf.cast(zeros, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred_sub = tf.where(masked, y_pred, zeros)
    y_true_sub = tf.where(masked, y_true, zeros)
    
    y_pred_rest = tf.where(masked2, y_pred, zeros)
    y_true_rest = tf.where(masked2, zeros, zeros)
    
    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    #print (sess.run(slice_y_greater_than_one)) # [2 4]
    #    print (sess.run(new_tensor)) # [2 4]


    contact = tf.reduce_sum(K.cast(K.equal(y_true_sub, 1), tf.int32))/2
    non_contact = tf.reduce_sum(K.cast(K.equal(y_true_sub, 0), tf.int32))/2
    y_true_sub = K.clip(y_true_sub, K.epsilon(), 1-K.epsilon())
    y_pred_sub = K.clip(y_pred_sub, K.epsilon(), 1-K.epsilon())
    weight = contact * loss_ratio
    weight = K.clip(weight, 1, 100000)
    if not y_true_sub.dtype == tf.float32:
        y_true_sub = K.cast(y_true_sub, tf.float32)
    if not y_pred_sub.dtype == tf.float32:
        y_pred_sub = K.cast(y_pred_sub, tf.float32)
    
    if not y_true_rest.dtype == tf.float32:
        y_true_rest = K.cast(y_true_rest, tf.float32)
    if not y_pred_rest.dtype == tf.float32:
        y_pred_rest = K.cast(y_pred_rest, tf.float32)
    if not weight.dtype == tf.float32:
        weight = K.cast(weight, tf.float32)
    logloss = -(y_true_sub * K.log(y_pred_sub)* weight + (1 - y_true_sub) * K.log(1 - y_pred_sub)* weight/2) - ((1 - y_true_rest) * K.log(1 - y_pred_rest))
    return K.mean( logloss, axis=-1)


class RnaGenerator_augment_gcn_torch(): 
    pass 

class RnaGenerator_augment_gcn_torch(Dataset):
    def __init__(self, dataset, batch_size, expected_n_channels, feature_type, include_pseudoknots):
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.feature_type = feature_type 
        self.include_pseudoknots = include_pseudoknots
        # sort table by length
        self.dataset = dataset.sort_values(["Length"], ascending=True).reset_index(drop=True)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        batch_data = self.dataset.iloc[index * self.batch_size: (index + 1) * self.batch_size] 
        [X,EE_val,EE_adj], [Y, nt_Y ] = get_input_output_rna_augment_bin_ntApairRegularized_gcn(batch_data, self.expected_n_channels, feature_type = self.feature_type, include_pseudoknots= self.include_pseudoknots)
        return [X,EE_val,EE_adj], [Y, nt_Y ]



class RnaGenerator_augment_gcn(Sequence):
    def __init__(self, dataset, batch_size, expected_n_channels, feature_type, include_pseudoknots):
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.feature_type = feature_type 
        self.include_pseudoknots = include_pseudoknots

        # sort table by length
        self.dataset = dataset.sort_values(["Length"], ascending=True).reset_index(drop=True)

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.dataset))
        #np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        batch_data = self.dataset.iloc[index * self.batch_size: (index + 1) * self.batch_size] # select rows of dataframe
        
        X,EE_val,EE_adj, Y, nt_Y = get_input_output_rna_augment_bin_ntApairRegularized_gcn(batch_data, self.expected_n_channels, feature_type = self.feature_type, include_pseudoknots= self.include_pseudoknots)
        
        return [X,EE_val,EE_adj], [Y, nt_Y ]
        

def get_feature_and_y_ntApairRegularized_gcn(batch_data, i,include_pseudoknots, fea_type = 0 ):

        #X, E_val, E_adj,Y0,nt_Y0,pair_Y0 = get_feature_and_y_ntApairRegularized_gcn(rna, all_ct_paths, expected_n_channels, fea_type = feature_type)
        sequence = batch_data['Sequence'][i]
    
        if include_pseudoknots:
            pairing_list = batch_data['BasePairs'][i].split(',')
        else:
            pairing_list = batch_data['UnknottedPairs'][i].split(',')
        
        linearProb = batch_data['LinearPartition'][i]
        #seqLen, node_feature, edge_val_feature, edge_feature,label,nt_label, pair_label, label_mask = extract_single_ntApairRegularized_gcn(ct_file, fea_type = fea_type)
        #return seqLen, node_feature, edge_val_feature, edge_feature, true_contact, nt_contact, pair_contact, label_mask
        ###########################################   Extract pairs
        nt_types = ['A','U','C','G']
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
        
        one_hot_feat = one_hot(sequence)
        seqLen = len(sequence)
        
        encode_feature = one_hot_feat
        
        label_mask = l_mask(one_hot_feat, seqLen)
    
        true_contact = np.zeros((seqLen, seqLen))
        nt_contact = np.zeros((seqLen, seqLen))
        
        for i in range(0,seqLen):
            for j in range(0,seqLen):
                xx = 0
                if i == j:
                    xx = 0
                if str(i+1)+"-"+str(j+1) in pairing_list or str(j+1)+"-"+str(i+1) in pairing_list:
                    xx = 1 
                true_contact[i, j] = xx
                true_contact[j, i] = xx
                
                tt = 0
                if i == j:
                    tt = 0
                if sequence[i]+sequence[j] in include_pairs:
                    tt = 1 
                nt_contact[i, j] = tt
                nt_contact[j, i] = tt
    
        true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
        # i - j >= 2
        true_contact = np.tril(true_contact, k=-2) + np.triu(true_contact, k=2) # remove the diagnol contact
        true_contact = true_contact.astype(np.uint8)
    
        nt_contact[nt_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
        # i - j >= 2
        nt_contact = np.tril(nt_contact, k=-2) + np.triu(nt_contact, k=2) # remove the diagnol contact
        nt_contact = nt_contact.astype(np.uint8)
        '''
        ### mark pair regions?
        map_matrix = true_contact
        
        L = seqLen
        sample_list= []
        for a in range(4-L,L-4):
            sublist = np.flipud(map_matrix).diagonal(offset=a)  # Vertical flip
            if np.sum(sublist)==0:
                continue
            if a<0:
                col_start = 0
                row_start = L-1+a
            else:
                col_start = a
                row_start = L-1
            col_segment = [[col_start+i for i,value in it] for key,it in itertools.groupby(enumerate(sublist), key=operator.itemgetter(1)) if key > 0]
            row_segment = [[row_start-i for i,value in it] for key,it in itertools.groupby(enumerate(sublist), key=operator.itemgetter(1)) if key > 0]  
            
            #print(col_segment)
            #print(row_segment)
        
            sample_list += col_segment
            sample_list += row_segment
    
        sample_list = list(itertools.chain.from_iterable(sample_list))  
        pair_contact = np.copy(map_matrix).astype(np.float32)
        pair_contact[pair_contact == 0] = -1
        #print("pair_contact1: ",pair_contact)
        for i in range(len(pair_contact)):
            if i in sample_list:
                pair_contact[:,i][pair_contact[:,i] != 1] = 0
                pair_contact[i,:][pair_contact[i,:] != 1] = 0            
        
        true_contact = np.copy(pair_contact)
        '''
        
        if fea_type == 1 or fea_type == 2:
            ### add pair features 
            tetrapair = np.zeros((seqLen, seqLen, 1))
            for i in range(0,seqLen):
                for j in range(i+1,seqLen):
                    if abs(i-j)<=4:
                        continue
                    nt_pair = sequence[i]+sequence[j]
                    if i > 0 and i < seqLen-1 and j > 0 and j < seqLen-1:
                      nt_pairL = sequence[i-1]+sequence[j+1]
                      nt_pairR = sequence[i+1]+sequence[j-1]
                      if nt_pair in include_pairs and nt_pairL in include_pairs:
                          tetrapair[i, j] = 1
                          tetrapair[j, i] = 1
                          tetrapair[i-1, j+1] = 1
                          tetrapair[j+1, i-1] = 1
                      elif nt_pair in include_pairs and nt_pairR in include_pairs:
                          tetrapair[i, j] = 1
                          tetrapair[j, i] = 1
                          tetrapair[i+1, j-1] = 1
                          tetrapair[j-1, i+1] = 1
                    elif i < seqLen-1 and j > 0:
                      nt_pairR = sequence[i+1]+sequence[j-1]
                      if nt_pair in include_pairs and (nt_pairR in include_pairs):
                          tetrapair[i, j] = 1
                          tetrapair[j, i] = 1
                          tetrapair[i+1, j-1] = 1
                          tetrapair[j-1, i+1] = 1
                    elif i > 0 and j < seqLen-1:
                      nt_pairL = sequence[i-1]+sequence[j+1]
                      if nt_pair in include_pairs and (nt_pairL in include_pairs):
                          tetrapair[i, j] = 1
                          tetrapair[j, i] = 1
                          tetrapair[i-1, j+1] = 1
                          tetrapair[j+1, i-1] = 1    
                            
            petrapair = np.zeros((seqLen, seqLen, 1))
            for i in range(0,seqLen):
                for j in range(i+1,seqLen):
                    if abs(i-j)<=4:
                        continue
                    nt_pair = sequence[i]+sequence[j]
                    if i > 0 and i < seqLen-1 and j > 0 and j < seqLen-1:
                      nt_pairL = sequence[i-1]+sequence[j+1]
                      nt_pairR = sequence[i+1]+sequence[j-1]
                      if nt_pair in include_pairs and nt_pairL in include_pairs and nt_pairR in include_pairs:
                          petrapair[i, j] = 1
                          petrapair[j, i] = 1
                          petrapair[i-1, j+1] = 1
                          petrapair[j+1, i-1] = 1
                          petrapair[i+1, j-1] = 1
                          petrapair[j-1, i+1] = 1
                          
            hetrapair = np.zeros((seqLen, seqLen, 1))
            for i in range(0,seqLen):
                for j in range(i+1,seqLen):
                    if abs(i-j)<=4:
                        continue
                    nt_pair = sequence[i]+sequence[j]
                    if i > 0 and i < seqLen-2 and j > 0 and j < seqLen-1:
                      nt_pairL = sequence[i-1]+sequence[j+1]
                      nt_pairR = sequence[i+1]+sequence[j-1]
                      nt_pairR2 = sequence[i+2]+sequence[j-2]
                      if nt_pair in include_pairs and nt_pairL in include_pairs and nt_pairR in include_pairs and nt_pairR2 in include_pairs:
                          hetrapair[i, j] = 1
                          hetrapair[j, i] = 1
                          hetrapair[i-1, j+1] = 1
                          hetrapair[j+1, i-1] = 1
                          hetrapair[i+1, j-1] = 1
                          hetrapair[j-1, i+1] = 1
                          hetrapair[i+2, j-2] = 1
                          hetrapair[j-2, i+2] = 1  
                      
            sixtrapair = np.zeros((seqLen, seqLen, 1))
            for i in range(0,seqLen):
                for j in range(i+1,seqLen):
                    if abs(i-j)<=4:
                        continue
                    nt_pair = sequence[i]+sequence[j]
                    if i > 0 and i < seqLen-3 and j > 0 and j < seqLen-1:
                      nt_pairL = sequence[i-1]+sequence[j+1]
                      nt_pairR = sequence[i+1]+sequence[j-1]
                      nt_pairR2 = sequence[i+2]+sequence[j-2]
                      nt_pairR3 = sequence[i+3]+sequence[j-3]
                      if nt_pair in include_pairs and nt_pairL in include_pairs and nt_pairR in include_pairs and nt_pairR2 in include_pairs and nt_pairR3 in include_pairs:
                          sixtrapair[i, j] = 1
                          sixtrapair[j, i] = 1
                          sixtrapair[i-1, j+1] = 1
                          sixtrapair[j+1, i-1] = 1
                          sixtrapair[i+1, j-1] = 1
                          sixtrapair[j-1, i+1] = 1
                          sixtrapair[i+2, j-2] = 1
                          sixtrapair[j-2, i+2] = 1
                          sixtrapair[i+3, j-3] = 1
                          sixtrapair[j-3, i+3] = 1
        
        if fea_type == -1:
            node_feature = encode_feature
            #edge_val_feature = true_contact
            #edge_feature = true_contact
            
            ## run linearPartition to get probability 
            # echo GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA | ~/tools/LinearPartition/linearpartition -V -p
            
            #outprob = ct_file+'.linearProb'
            #command = "echo "+sequence+" | /faculty/jhou4/tools/LinearPartition/linearpartition -V -r "+outprob
            #os.system(command)
            if linearProb[0] == ',':
                linearProb = linearProb[1:]
            if linearProb[0] == ';':
                linearProb = linearProb[1:]
            
            lines = linearProb.split(',')
            
            nt_pair_prob = np.zeros((seqLen, seqLen))
            
            for line in lines:
                arr = line.strip().split('-')
                if not line.startswith('#') and len(arr)==3:
                    #print(arr)
                    nt_index = int(arr[0])
                    pair_index = int(arr[1])
                    pair_prob = float(arr[2])
                    nt_pair_prob[nt_index-1,pair_index-1] = pair_prob
                    nt_pair_prob[pair_index-1,nt_index-1] = pair_prob
                    
            edge_feature = np.copy(nt_pair_prob)
            edge_feature[edge_feature>0.5]=1
            edge_val_feature = np.copy(edge_feature)  ## after testing, using 
            
            # should we set diagnal to 1?
            np.fill_diagonal(edge_feature, 1)
            
            # should we set diagnal to 1?
            np.fill_diagonal(edge_val_feature, 1)
            
            
        elif fea_type == 0:
            node_feature = encode_feature
            edge_val_feature = nt_contact
            edge_feature = nt_contact
        elif fea_type == 1:
            num_neighbors=3
            node_feature = encode_feature
            edge_val_feature = tetrapair + petrapair + hetrapair + sixtrapair
            edge_val_feature = edge_val_feature[:,:,0]/4
            
            edge_feature_pos = np.copy(edge_val_feature)
            edge_feature_pos[edge_feature_pos>1]=1
    
            edge_val_feature_inv =  np.exp(-edge_val_feature)  
            knns = np.argpartition(edge_val_feature_inv, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
            # Make connections 
            seqLen = len(edge_val_feature)
            edge_feature = np.zeros((seqLen, seqLen))
            for idx in range(seqLen):
                edge_feature[idx][knns[idx]] = 1
            
            edge_feature = edge_feature * edge_feature_pos
            
            ## add neighbors
            for idx in range(seqLen):
                if idx == 0:
                  edge_feature[idx][idx+1] = 1
                  edge_feature[idx+1][idx] = 1
                elif idx == seqLen-1:
                  edge_feature[idx][idx-1] = 1
                  edge_feature[idx-1][idx] = 1
                else:
                  edge_feature[idx][idx+1] = 1
                  edge_feature[idx+1][idx] = 1
                  edge_feature[idx][idx-1] = 1
                  edge_feature[idx-1][idx] = 1
                
            #np.fill_diagonal(edge_feature, 2)  # Special token for self-connections
        elif fea_type == 2:
            num_neighbors=3
            node_feature = encode_feature
            edge_val_feature = tetrapair + petrapair + hetrapair + sixtrapair
            edge_val_feature = edge_val_feature[:,:,0]/4
            
            edge_feature_pos = np.copy(edge_val_feature)
            edge_feature_pos[edge_feature_pos>1]=1
    
            edge_val_feature_inv =  np.exp(-edge_val_feature)  
            knns = np.argpartition(edge_val_feature_inv, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
            # Make connections 
            seqLen = len(edge_val_feature)
            edge_feature = np.zeros((seqLen, seqLen))
            '''
            for idx in range(seqLen):
                edge_feature[idx][knns[idx]] = 1
            '''
            
            for idx in range(seqLen):
                for col in knns[idx]:
                    if edge_val_feature[idx][col] != 0: # avoid the neighbors < 3, avoid randomly choose 0
                        edge_feature[idx][col] = 1
                        edge_feature[col][idx] = 1
            
            edge_feature = edge_feature * edge_feature_pos
            
            ## add neighbors
            '''
            for idx in range(seqLen):
                if idx == 0:
                  edge_feature[idx][idx+1] = 1
                  edge_feature[idx+1][idx] = 1
                elif idx == seqLen-1:
                  edge_feature[idx][idx-1] = 1
                  edge_feature[idx-1][idx] = 1
                else:
                  edge_feature[idx][idx+1] = 1
                  edge_feature[idx+1][idx] = 1
                  edge_feature[idx][idx-1] = 1
                  edge_feature[idx-1][idx] = 1
            '''
            #np.fill_diagonal(edge_feature, 2)  # Special token for self-connections
            
            ### add physicochemical_property_indices (L,L,8)
            physicochemical = np.zeros((seqLen, seqLen, 22))
            for i in range(0,seqLen):
                for j in range(0,seqLen):
                    nt_pair = sequence[i]+sequence[j]
                    if nt_pair in physicochemical_property_indices and edge_feature_pos[i,j]>0:
                        physicochemical[i, j] = physicochemical_property_indices[nt_pair] 
                    else:
                        physicochemical[i, j] = 0
                        #raise Exception(nt_pair,' is not found in physicochemical_property_indices')
    
            edge_val_feature = physicochemical
            
            # should we set diagnal to 1?
            np.fill_diagonal(edge_feature, 1)     
            
            # should we set offset diagnal to 1?
            rng = np.arange(len(edge_feature)-1)
            edge_feature[rng, rng+1] = 1
            edge_feature[rng+1, rng] = 1
            
            
        elif fea_type == 3:
            node_feature = encode_feature
            #edge_val_feature = true_contact
            #edge_feature = true_contact
            
            ## run linearPartition to get probability 
            # echo GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA | ~/tools/LinearPartition/linearpartition -V -p
            
            #outprob = ct_file+'.linearProb'
            #command = "echo "+sequence+" | /faculty/jhou4/tools/LinearPartition/linearpartition -V -r "+outprob
            #os.system(command)
            
            if linearProb[0] == ',':
                linearProb = linearProb[1:]
            if linearProb[0] == ';':
                linearProb = linearProb[1:]
            
            lines = linearProb.split(',')
            
            nt_pair_prob = np.zeros((seqLen, seqLen))
            
            for line in lines:
                arr = line.strip().split('-')
                if not line.startswith('#') and len(arr)==3:
                    #print(arr)
                    nt_index = int(arr[0])
                    pair_index = int(arr[1])
                    pair_prob = float(arr[2])
                    nt_pair_prob[nt_index-1,pair_index-1] = pair_prob
                    nt_pair_prob[pair_index-1,nt_index-1] = pair_prob
                    
            edge_feature = np.copy(nt_pair_prob)
            edge_feature[edge_feature>0.5]=1
            
            # should we set diagnal to 1?
            #np.fill_diagonal(edge_val_feature, 1)
                
            #np.fill_diagonal(edge_feature, 2)  # Special token for self-connections
                
            #np.fill_diagonal(edge_feature, 2)  # Special token for self-connections
            
            ### add physicochemical_property_indices (L,L,8)
            physicochemical = np.zeros((seqLen, seqLen, 22))
            for i in range(0,seqLen):
                for j in range(0,seqLen):
                    nt_pair = sequence[i]+sequence[j]
                    if nt_pair in physicochemical_property_indices:
                        physicochemical[i, j] = physicochemical_property_indices[nt_pair] 
                    else:
                        physicochemical[i, j] = 0
                        #raise Exception(nt_pair,' is not found in physicochemical_property_indices')
    
            edge_val_feature = physicochemical
            
            #edge_val_feature = 1/(1 + np.exp(-edge_val_feature))
            
            # should we set diagnal to 1?
            np.fill_diagonal(edge_feature, 1)        
       
            # should we set offset diagnal to 1?
            rng = np.arange(len(edge_feature)-1)
            edge_feature[rng, rng+1] = 1
            edge_feature[rng+1, rng] = 1
            
    
        ###########################################   Extract pairs    

        # Create X and Y placeholders
        # Sequence features
        X = np.full((seqLen, expected_n_channels), 0)
        X[:, 0:expected_n_channels] = node_feature

        if fea_type == 0 or fea_type == 1 or fea_type == -1:
            E_val = np.full((seqLen, seqLen, 1), 0)
            E_val[:, :, 0] = edge_val_feature
        elif fea_type == 2 or fea_type == 3:
            E_val = np.full((seqLen, seqLen, 22), 0)
            E_val[:, :, 0:22] = edge_val_feature
        
        E_adj = np.full((seqLen, seqLen), 0)
        E_adj[:, :] = edge_feature
    
    
        Y0 = np.full((seqLen, seqLen), 0)
        nt_Y0 = np.full((seqLen, seqLen), 0)
    
        # label
        Y0[:, :] = true_contact
        
        # nt label
        nt_Y0[:, :] = nt_contact
        return X, E_val, E_adj, Y0, nt_Y0


def get_input_output_rna_augment_bin_ntApairRegularized_gcn(batch_data, expected_n_channels, feature_type, include_pseudoknots):
    # get maximum length
    OUTL = batch_data["Length"].max()

    #### find the dimension
    total_dim = len(batch_data)
    
    #### Define node matrix
    XX = np.full((total_dim, OUTL, expected_n_channels), 0)
    
    #### Define edge matrix
    if feature_type == 0 or feature_type == 1 or feature_type == -1:
        EE_val = np.full((total_dim, OUTL, OUTL, 1), 0)
    elif feature_type == 2 or feature_type == 3:
        EE_val = np.full((total_dim, OUTL, OUTL, 22), 0)
    
    
    EE_adj = np.full((total_dim, OUTL, OUTL), 0)
    
    
    #### Define output
    
    YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    nt_YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    pair_YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    
    
    #print("Min: ",L_min, " Max: ", OUTL, " Final: ", XX.shape)
    indx = 0
    
    for i in batch_data.index:
        rna = batch_data['RNA_ID'][i]
        
        X, E_val, E_adj,Y0,nt_Y0 = get_feature_and_y_ntApairRegularized_gcn(batch_data, i, include_pseudoknots,  fea_type = feature_type)

        assert len(X[0, :]) == expected_n_channels
        assert len(X[:, 0]) >= len(Y0[:, 0])
        if len(X[:, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        
        l = len(X[:, 0])
        XX[indx, :l, :] = X 
        EE_val[indx, :l, :l, :] = E_val
        EE_adj[indx, :l, :l] = E_adj
        
        YY[indx, :l, :l, 0] = Y0
        nt_YY[indx, :l, :l, 0] = nt_Y0
        indx += 1
  
    return XX.astype(np.float32), EE_val.astype(np.float32), EE_adj.astype(np.float32), YY.astype(np.float32), nt_YY.astype(np.float32)


#https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras

def tp(y_true, y_pred):
    #y_true = tf.reshape(y_true, [-1])
    #y_pred = tf.reshape(y_pred, [-1])
    #y_true_shape = K.print_tensor(y_true.get_shape(), message='y_true = ')
    #y_pred_shape = K.print_tensor(y_pred.get_shape(), message='y_pred = ')
    true_positives = tf.reduce_sum(tf.cast(tf.greater(y_true * y_pred, 0.5), tf.int32))/2
    possible_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.int32))/2
    return true_positives

def pp(y_true, y_pred):
    #y_true = tf.reshape(y_true, [-1])
    #y_pred = tf.reshape(y_pred, [-1])
    true_positives = tf.reduce_sum(tf.cast(tf.greater(y_true * y_pred, 0.5), tf.int32))/2
    possible_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.int32))/2
    return possible_positives

def num_contact(y_true, y_pred):
    contact = tf.reduce_sum(K.cast(K.equal(y_true, 1), tf.int32))/2
    non_contact = tf.reduce_sum(K.cast(K.equal(y_true, 0), tf.int32))/2
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    weight = contact
    if not weight.dtype == tf.float32:
        weight = K.cast(weight, tf.float32)
    return weight

def data_size(y_true, y_pred):
    size =K.shape(y_true)
    return size

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def fbeta(y_true, y_pred, beta=2):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)

#matthews_correlation_coefficient
def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def equal_error_rate(y_true, y_pred):
    n_imp = tf.math.count_nonzero(tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_gen = tf.math.count_nonzero(tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    scores_imp = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
    scores_gen = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    cond = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.math.count_nonzero(tf.greater_equal(scores_imp, t), dtype=tf.float32), n_imp),
        tf.divide(tf.math.count_nonzero(tf.less(scores_gen, t), dtype=tf.float32), n_gen)
    )
    t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2

    return eer



# ------------- one hot encoding of RNA sequences -----------------#
def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    # non standard nt is marked as [-1, -1, -1, -1]
    return feat

def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)



### evaluation

def evaluate_predictions_single(reference_matrix,pred_matrix,sequence,label_mask, verbose=False):
    #print("reference_structure.shape: ",reference_matrix.shape)
    #print("predicted_structure.shape: ",pred_matrix.shape)
    reference_structure = get_ss_pairs_from_matrix(reference_matrix,sequence,label_mask, 0.5)
    predicted_structure = get_ss_pairs_from_matrix(pred_matrix,sequence,label_mask, 0.5)
    if verbose:
      print("reference_structure len: ",len(reference_structure))
      print("reference_structure: ",reference_structure)
      print("predicted_structure len: ",len(predicted_structure))
      print("predicted_structure: ",predicted_structure)
    acc = compare_structures('N'*len(reference_matrix),reference_structure,predicted_structure)
    
    return acc


# Compare predicted structure against ref. Calculate similarity statistics
def compare_structures(ref_seq,pred, ref, verbose = 0):
    pred_p = pred_n = tp = tn = 0
    # Count up how many SS and DS predictions were correct, relative to the phylo structure
    n_pairs = len(ref_seq)
    for i in range(1, n_pairs+1):
        for j in range(i+1, n_pairs+1):
            pair = str(i)+'-'+str(j)
            if pair in pred:
                pred_p += 1
                if pair in ref:
                    tp += 1
            else:
                pred_n += 1
                if pair not in ref:
                    tn += 1
    fp = pred_p - tp
    fn = pred_n - tn
    fp = float(fp)
    tp = float(tp)
    fn = float(fn)
    fp = float(fp)
    ppv 		= round((100 * tp) / (tp + fp+ sys.float_info.epsilon), 2)
    sensitivity	= round((100 * tp) / (tp + fn+ sys.float_info.epsilon), 2)
    npv			= round((100 * tn) / (tn + fn+ sys.float_info.epsilon), 2)
    specificity	= round((100 * tn) / (tn + fp+ sys.float_info.epsilon), 2)
    accuracy 	= round((100 * (tp + tn)) / (tp + fp + tn + fn+ sys.float_info.epsilon), 2)
    
    f1_score = round((2 * ppv * sensitivity) / (ppv + sensitivity+ sys.float_info.epsilon),2)
    up = tp*tn - fp*fn
    down = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    
    mcc = up / (down + sys.float_info.epsilon)*100
    mcc = round(mcc,2)
    #mcc = np.where(np.isnan(mcc), np.zeros_like(mcc), mcc)
    
    
    
    #print("")
    #print("    Results with respect to SS")
    #print("    ---------------------------------------")
    #print("")
    #print("    Positive predictive value: "+str(ppv))
    #print("    Sensitivity:               "+str(sensitivity))
    #print("    Negative predictive value: "+str(npv))
    #print("    Specificity:               "+str(specificity))
    #print("    F-score:          "+str(f1_score))
    #print("    MCC:          "+str(mcc))
    #print("    Overall accuracy:          "+str(accuracy))
    #print("")
    if verbose != 0:
        print("Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy")
        print("Results with respect to SS :     %.3f        %.3f        %.3f      %.3f      %.3f      %.3f      %.3f" \
            %(ppv, sensitivity, npv, specificity, f1_score, mcc,accuracy))


    
    return [ppv, sensitivity, npv, specificity, f1_score, mcc,accuracy]

def output_mask(seq, NC=True):
    seq = seq.upper()
    if NC:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
    else:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
    mask = np.zeros((len(seq), len(seq)))
    for i, I in enumerate(seq):
        for j, J in enumerate(seq):
            if str(I) + str(J) in include_pairs:
                mask[i, j] = 1
    return mask

def get_ss_pairs_from_matrix(pair_matrix,sequence,label_mask,Threshold):
    ones = np.ones((len(pair_matrix), len(pair_matrix)))
    #test_output= pair_matrix[np.triu(ones, 2) == 1][..., np.newaxis]
    test_output= np.triu(pair_matrix, 2)
    mask = output_mask(sequence, NC=flag_noncanonical)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    #for i in range(test_output.shape[0]):
    for i in range(len(inds[0])):
        y_pred[inds[0][i], inds[1][i]] = test_output[inds[0][i]][inds[1][i]]
    y_pred = np.multiply(y_pred, mask)
    
    tri_inds = np.triu_indices(y_pred.shape[0], k=1)

    out_pred = y_pred[tri_inds]
    outputs = out_pred[:, None]
    
    #sequence = 'N'*seqLen
    
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([sequence[tri_inds[0][j]], sequence[tri_inds[1][j]]])] for j in
                 range(tri_inds[0].shape[0])]
    #print(seq_pairs)
    outputs_T = np.greater_equal(outputs, Threshold)
    #print("outputs_T.sum: ",outputs_T)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    #print("pred_pairs1: ",pred_pairs)
    pred_pairs = [i[:2] for i in pred_pairs]
    #print("pred_pairs2: ",pred_pairs)
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred) # remove temporary
    #print("pred_pairs3: ",pred_pairs)
    pred_pairs_pred = [str(i[0]+1)+'-'+str(i[1]+1) for i in pred_pairs]
    return pred_pairs_pred


# ----------------------- find multiplets pairs--------------------------------#
def multiplets_pairs(pred_pairs):

    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)
        
    return temp3

def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs)+len(save_multiplets)
    #print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets



def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def output_result_simple(avg_acc):
    if len(avg_acc) == 0:
        avg_acc = [0,0,0,0,0,0,0]
    print("Range(> 2):")
    print("Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy")
    print("Acc:     %.3f        %.3f        %.3f      %.3f      %.3f      %.3f      %.3f" \
            %(avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3],avg_acc[4],avg_acc[5],avg_acc[6]))
    

def ct_file_output(pairs, seq, save_result_path):

    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)
    for i, I in enumerate(pairs):
        arr = I.split('-')
        col5[int(arr[0])-1] = int(arr[1])
        col5[int(arr[1])-1] = int(arr[0])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(save_result_path, (temp), delimiter='\t', fmt="%s", header=str(len(seq)) + '\t\t' + 'Prediction' + '\t\t' + 'output' , comments='')

    return
