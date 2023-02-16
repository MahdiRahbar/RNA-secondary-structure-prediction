# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0

import sys 
import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


import datetime
import argparse
import random
import operator
import itertools
import operator
flag_plots = False


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from time import sleep
from tqdm import tqdm

from model import * 
from preprocess import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)



if __name__ == "__main__":


    ######################################################################### Define argument

    args = get_args()

    #file_weights              = args.file_weights #dir_out+'/rna.hdf5' 
    batch_size                = args.batch_size #5
    dev_size                  = args.dev_size #  100000 
    training_window           = args.training_window #64  
    training_epochs           = args.training_epochs #10  
    arch_depth                = args.arch_depth #4 
    filters_per_layer         = args.filters_per_layer #8
    len_range                 = args.len_range #0-50
    data_path               = args.data_path #'/faculty/jhou4/Projects/RNA_folding/data/Own_data/Sharear' 
    dir_out                   = args.dir_out #'/faculty/jhou4/Projects/RNA_folding/train_results_20201128' 
    filter_size_2d            = args.filter_size_2d
    flag_eval_only            = False
    flag_noncanonical    = False
    if args.flag_eval_only == 1:
        flag_eval_only = True

    if args.flag_noncanonical == 1:
        flag_noncanonical = True

    length_start              = 0
    length_end                = 50
    if len(len_range.split('-')) == 2:
        length_start              = int(len_range.split('-')[0])
        length_end                = int(len_range.split('-')[1])
    pad_size                  = 0

    file_weights              = dir_out+'/rna.hdf5' 

    loss_ratio                = args.loss_ratio # 1

    dropout_rate              = args.dropout_rate # 1
    lstm_layers               = args.lstm_layers # 1
    fully_layers              = args.fully_layers # 1
    nt_reg_weight               = args.nt_reg_weight # 1
    pair_reg_weight               = args.pair_reg_weight # 1
    lstm_filter     = args.lstm_filter # 1
    feature_type     = args.feature_type # 1
    dilation_size =  args.dilation_size # 1
    regularize            = False
    if args.weight_regularization == 1:
        regularize = True
        print("Activating regularization")

    if args.include_pseudoknots == 1:
        include_pseudoknots = True
        print("Including pseudoknots")
    else:
        include_pseudoknots = False
        print("Excluding pseudoknots")

    if feature_type == 0 or feature_type == -1:
        expected_n_channels       = 4
    elif feature_type == 1:
        expected_n_channels       = 4
    elif feature_type == 2:
        expected_n_channels       = 4
    elif feature_type == 3:
        expected_n_channels       = 4
    else:
        print("Wrong ",feature_type)
        exit(-1)

    print('Start ' + str(datetime.datetime.now()))

    print('')
    print('Parameters:')
    print('dev_size', dev_size)
    print('file_weights', file_weights)
    print('training_window', training_window)
    print('training_epochs', training_epochs)
    print('arch_depth', arch_depth)
    print('filters_per_layer', filters_per_layer)
    print('pad_size', pad_size)
    print('batch_size', batch_size)
    print('data_path', data_path)
    print('dir_out', dir_out)
    print('flag_noncanonical', flag_noncanonical)

    os.system('mkdir -p ' + dir_out)





    # #########################################################################  2. Define the model


    # Allow GPU memory growth
    if hasattr(tf, 'GPUOptions'):
        from tensorflow.python.keras import backend as K
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #K.tensorflow_backend.set_session(sess)
        K.set_session(sess)
    else:
        # For other GPUs
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    print('')
    print('Build a model..')
    model = ''
    #arch_depth=4
    model = rna_pair_prediction_bin_gcn5(num_gcn_layers = arch_depth, filter_size = filter_size_2d, num_lstm_layers = lstm_layers, hidden_dim = filters_per_layer, regularize=regularize, dropout_rate = dropout_rate, dilation_size=1)


    print('')
    print('Compile model..')



    # define two dictionaries: one that specifies the loss method for
    # each output of the network along with a second dictionary that
    # specifies the weight per loss
    #losses = {
    #	"pair_out": weighted_binary_crossentropy,
    #	"nt_out": weighted_binary_crossentropy_ntRegularized,
    #  "pair_out2": weighted_binary_crossentropy_pairRegularized
    #}


    losses = {
        "nt_out": weighted_binary_crossentropy_ntRegularized,
    "pair_out2": weighted_binary_crossentropy_pairRegularized
    }
    '''

    losses = {
        "nt_out": weighted_binary_crossentropy_ntRegularized,
    "pair_out2": tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
    }
    '''

    #opt = tf.keras.optimizers.Adam(lr=1e-4)
    opt = tf.keras.optimizers.Adam(0.001, decay=2.5e-4)

    #lossWeights = {"pair_out": 1.0, "pair_out2": pair_reg_weight, "nt_out": nt_reg_weight}
    lossWeights = {"pair_out2": pair_reg_weight, "nt_out": nt_reg_weight}

    model.compile(
        optimizer=opt, # you can use any other optimizer
        loss = losses, #loss='binary_crossentropy',
        loss_weights=lossWeights,
        metrics=[
            #"accuracy",
            precision,
            recall,
            f1,
            #fbeta,
            specificity,
            #negative_predictive_value,
            mcc,
            tp,
            pp,
            #equal_error_rate,
            #num_contact,
            data_size        
        ]
    )

    print(model.summary(line_length=150))



    ######################################################################### 3. Define the dataset 

    seq_len_range = (length_start,length_end)
    dataset = pd.read_hdf(data_path, "df").query("Length >= {} and Length <= {}".format(*seq_len_range)).reset_index(drop=True)

    dataset['StructureEnergy'] = dataset['StructureEnergy'].astype(float)


    ### Load bpRNA data
    bpRNA_train_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Train') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)
    bpRNA_valid_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Validation') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)
    bpRNA_test_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Test') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)

    ### Load PDB data
    pdb_train_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Train') )].reset_index(drop=True)
    pdb_valid_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Validation'))].reset_index(drop=True)
    pdb_test_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test') )].reset_index(drop=True)
    pdb_test2_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test2'))].reset_index(drop=True)
    pdb_test3_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test_hard'))].reset_index(drop=True)


    bpRNA_train_generator = RnaGenerator_augment_gcn(bpRNA_train_data, batch_size, expected_n_channels)
    bpRNA_valid_generator = RnaGenerator_augment_gcn(bpRNA_valid_data, batch_size, expected_n_channels)

    PDB_train_generator = RnaGenerator_augment_gcn(pdb_train_data, batch_size, expected_n_channels)
    PDB_valid_generator = RnaGenerator_augment_gcn(pdb_valid_data, batch_size, expected_n_channels)


    print('')
    print('len(bpRNA_train_generator) : ' + str(len(bpRNA_train_generator)))
    print('len(bpRNA_valid_generator) : ' + str(len(bpRNA_valid_generator)))

    [X,EE_val,EE_adj], [Y, nt_Y] = bpRNA_train_generator[1]

    print('Actual shape of X    : ' + str(X.shape))
    print('Actual shape of EE_val    : ' + str(EE_val.shape))
    print('Actual shape of EE_adj    : ' + str(EE_adj.shape))
    print('Actual shape of Y    : ' + str(Y.shape))
    print('Actual shape of nt_Y    : ' + str(nt_Y.shape))

    print("Y==1: ", np.count_nonzero(Y == 1))
    print("Y==0: ", np.count_nonzero(Y == 0))
    print("Y==-1: ", np.count_nonzero(Y == -1))



    eva_dataset = pd.concat([bpRNA_valid_data, bpRNA_test_data,pdb_train_data,pdb_valid_data,pdb_test_data,pdb_test2_data,pdb_test3_data], ignore_index=True).reset_index(drop=True)

    print('')
    #print('Channel summaries:')
    #summarize_channels(X[0, :, :], Y[0])

    if flag_eval_only == 0:
        if os.path.exists(file_weights):
            print('')
            print('Loading existing weights..')
            try:
                model.load_weights(file_weights)
            except:
                print("Loading model error!")
        print('')
        print('Train..')
        
        with open(dir_out+'/bpRNA_validation_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()
        
        with open(dir_out+'/bpRNA_test_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()
        
        with open(dir_out+'/PDB_train_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()
        
        with open(dir_out+'/PDB_validation_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()

        with open(dir_out+'/PDB_test_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()

        with open(dir_out+'/PDB_test2_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()

        with open(dir_out+'/PDB_test_hard_process.txt',"w") as newFile:
            header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % ("Epoch", "ppv", "sensitivity", "npv", "specificity", "f1_score", "mcc", "accuracy")
            newFile.write(header)
        newFile.close()
        
        
        for epoch in range(training_epochs):
            
            print("\n################### Training on bpRNA dataset ###################\n")
            bpRNA_history = model.fit_generator(generator = bpRNA_train_generator,
                validation_data = bpRNA_valid_generator,
                callbacks = [ModelCheckpoint(filepath = file_weights, mode='max', monitor = 'val_pair_out2_f1', save_best_only = True, save_weights_only = True, verbose = 1)],
                verbose = 1,
                max_queue_size = 8,
                workers = 1,
                use_multiprocessing = False,
                shuffle = True ,
                epochs = 2)
            
            print("\n################### Training on PDB dataset ###################\n")
            PDB_history = model.fit_generator(generator = PDB_train_generator,
                validation_data = PDB_valid_generator,
                callbacks = [ModelCheckpoint(filepath = file_weights, mode='max', monitor = 'val_pair_out2_f1', save_best_only = True, save_weights_only = True, verbose = 1)],
                verbose = 1,
                max_queue_size = 8,
                workers = 1,
                use_multiprocessing = False,
                shuffle = True ,
                epochs = 2)
            
            
            model.save(dir_out+'/rna_epoch_'+str(epoch)+'.hdf5' )  # creates a HDF5 file 'my_model.h5'
            
            # evaluate

            bpRNA_acc_record_val=[]
            bpRNA_acc_record_test=[]
            pdb_acc_record_train=[]     
            pdb_acc_record_val=[]    
            pdb_acc_record_test=[]        
            pdb_acc_record_test2=[]     
            pdb_acc_record_test3=[]      
            print("\n\nEvaluation on Epoch: " + str(int(epoch))+"\n")
            
            print("Evaluating on all data with ", len(eva_dataset), " rnas\n")
            length2index = {}
            for i in eva_dataset.index:
                seqLen = eva_dataset['Length'][i]
                if seqLen in length2index:
                    length2index[seqLen].append(i)
                else:
                    length2index[seqLen] = [i]
            idx = 0
            for id_length in length2index.keys():
                rna_idxs = length2index[id_length]
                X_all = []
                E_val_all = []
                E_adj_all = []
                for i in rna_idxs:
                    idx += 1
                    X, E_val, E_adj,Y0,nt_Y0 = get_feature_and_y_ntApairRegularized_gcn(eva_dataset, i, fea_type = feature_type)
                    
                    #X, E_val, E_adj,Y0,nt_Y0 = get_feature_and_y_ntApairRegularized_gcn(batch_data, i, fea_type = feature_type)
                            
                    X = np.expand_dims(X, axis=0)
                    E_val = np.expand_dims(E_val, axis=0)
                    E_adj = np.expand_dims(E_adj, axis=0)
                    X_all.append(X)
                    E_val_all.append(E_val)
                    E_adj_all.append(E_adj)
                
                X_all_np = np.concatenate(X_all, axis = 0)
                E_val_all_np = np.concatenate(E_val_all, axis = 0)
                E_adj_all_np = np.concatenate(E_adj_all, axis = 0)
                #print("X_all_np: ",X_all_np.shape)
                model_pred = rna_pair_prediction_bin_gcn5(num_gcn_layers = arch_depth, filter_size = filter_size_2d, num_lstm_layers = lstm_layers, hidden_dim = filters_per_layer, regularize=regularize, dropout_rate = dropout_rate, dilation_size=1)
            
                model_pred.load_weights(dir_out+'/rna_epoch_'+str(epoch)+'.hdf5')                
                output_prob_ids, nt_output_prob_ids = model_pred.predict([X_all_np,E_val_all_np,E_adj_all_np], batch_size = 5)
                #print("output_prob_ids: ",output_prob_ids.shape)
                
                print(str(idx)+",", end="", flush=True)
            
                idx_num = 0
                for rna_id in rna_idxs:
                    rna = eva_dataset['RNA_ID'][rna_id]
                    rna_datasource = eva_dataset['DataSource'][rna_id]
                    rna_datatype = eva_dataset['DataType'][rna_id]
                    sequence = eva_dataset['Sequence'][rna_id]
                    if include_pseudoknots:
                        pairing_list = eva_dataset['BasePairs'][rna_id].split(',')
                    else:
                        pairing_list = eva_dataset['UnknottedPairs'][rna_id].split(',')
                    
            
                    one_hot_feat = one_hot(sequence)
                    label_mask = l_mask(one_hot_feat, len(sequence))
                    
                    output_prob = output_prob_ids[idx_num]
                    idx_num += 1
                    output_class = output_prob > 0.5
                    seqLen = len(sequence)
                    true_contact = np.zeros((seqLen, seqLen))
                    for i in range(0,seqLen):
                        for j in range(0,seqLen):
                            xx = 0
                            if i == j:
                                xx = 0
                            if str(i+1)+"-"+str(j+1) in pairing_list or str(j+1)+"-"+str(i+1) in pairing_list:
                                xx = 1 
                            true_contact[i, j] = xx
                            true_contact[j, i] = xx
                
                    true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
                    # i - j >= 2
                    true_contact = np.tril(true_contact, k=-2) + np.triu(true_contact, k=2) # remove the diagnol contact
                    true_contact = true_contact.astype(np.uint8)
            
                    acc = evaluate_predictions_single(true_contact, output_prob[:,:,0],sequence,label_mask)
                    
                    if rna_datasource == 'bpRNA' and rna_datatype == 'Validation':
                        bpRNA_acc_record_val.append(acc) 
                    if rna_datasource == 'bpRNA' and rna_datatype == 'Test':
                        bpRNA_acc_record_test.append(acc) 
                    if rna_datasource == 'PDB' and rna_datatype == 'Train':
                        pdb_acc_record_train.append(acc) 
                    if rna_datasource == 'PDB' and rna_datatype == 'Validation':
                        pdb_acc_record_val.append(acc) 
                    if rna_datasource == 'PDB' and rna_datatype == 'Test':
                        pdb_acc_record_test.append(acc) 
                    if rna_datasource == 'PDB' and rna_datatype == 'Test2':
                        pdb_acc_record_test2.append(acc) 
                    if rna_datasource == 'PDB' and rna_datatype == 'Test_hard':
                        pdb_acc_record_test3.append(acc) 
            
            print("\n\n########################### Epoch: ",int(epoch)," ################################")

            
            print("\n\n########################### bpRNA validation evaluation (total ",len(bpRNA_valid_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(bpRNA_acc_record_val), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/bpRNA_validation_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
            print("\n\n########################### bpRNA test evaluation (total ",len(bpRNA_test_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(bpRNA_acc_record_test), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/bpRNA_test_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
            print("\n\n########################### PDB train evaluation (total ",len(pdb_train_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(pdb_acc_record_train), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/PDB_train_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
            print("\n\n########################### PDB validation evaluation (total ",len(pdb_valid_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(pdb_acc_record_val), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/PDB_validation_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
            
            print("\n\n########################### PDB test evaluation (total ",len(pdb_test_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(pdb_acc_record_test), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/PDB_test_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
            print("\n\n########################### PDB test2 evaluation (total ",len(pdb_test2_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(pdb_acc_record_test2), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/PDB_test2_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
            print("\n\n########################### PDB test_hard evaluation (total ",len(pdb_test3_data)," rnas): ###################################\n\n")
            avg_acc = np.mean(np.array(pdb_acc_record_test3), axis=0).round(decimals=5)
            output_result_simple(avg_acc)
            
            with open(dir_out+'/PDB_test_hard_process.txt',"a") as newFile:
                header = "%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n" % (str(int(epoch)), str(avg_acc[0]), str(avg_acc[1]), str(avg_acc[2]), str(avg_acc[3]), str(avg_acc[4]), str(avg_acc[5]), str(avg_acc[6]))
                newFile.write(header)
            newFile.close()
                
                            
            print("\n\n#######################################################################\n\n")
            
            print(" Name: ",rna," Length: ",str(len(output_class)))
            print("Prediction info:")
            print("output_class: ",output_class.sum(), " Pair_Labels: ",true_contact.sum(), " output_class.shape: ",output_class.shape, " output_prob.shape: ",output_prob.shape, " true_contact.shape: ",true_contact.shape)
            print("output_prob ranges: ",[np.percentile(output_prob[:,:,0], 0),np.percentile(output_prob[:,:,0], 25),np.percentile(output_prob[:,:,0], 50),np.percentile(output_prob[:,:,0], 75),np.percentile(output_prob[:,:,0], 100)])
