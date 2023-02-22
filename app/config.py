# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0


# args = get_args()
file_weights              = dir_out+'/rna.hdf5'  # args.file_weights #dir_out+'/rna.hdf5' 
batch_size                = 5 # args.batch_size #5
dev_size                  = 100000 # args.dev_size #  100000 
training_window           = 64 # args.training_window #64  
training_epochs           = 10 # args.training_epochs #10  
arch_depth                = 4 # args.arch_depth #4 
filters_per_layer         = 8 # args.filters_per_layer #8
len_range                 = "0-50" # args.len_range #0-50
data_path               = '/faculty/jhou4/Projects/RNA_folding/data/Own_data/Sharear'  # args.data_path #'/faculty/jhou4/Projects/RNA_folding/data/Own_data/Sharear' 
dir_out                   = '/faculty/jhou4/Projects/RNA_folding/train_results_20201128'  # args.dir_out #'/faculty/jhou4/Projects/RNA_folding/train_results_20201128' 
filter_size_2d            = 4 # args.filter_size_2d
flag_eval_only            = False
flag_noncanonical    = False
# if args.flag_eval_only == 1:
#     flag_eval_only = True

# if args.flag_noncanonical == 1:
#     flag_noncanonical = True

length_start              = 0
length_end                = 50
if len(len_range.split('-')) == 2:
    length_start              = int(len_range.split('-')[0])
    length_end                = int(len_range.split('-')[1])
pad_size                  = 0

file_weights              = dir_out+'/rna.hdf5' 

loss_ratio                = 1 # args.loss_ratio # 1

dropout_rate              = 1 # args.dropout_rate # 1
lstm_layers               = 1 # args.lstm_layers # 1
fully_layers              = 1 # args.fully_layers # 1
nt_reg_weight               = 1 # args.nt_reg_weight # 1
pair_reg_weight               = 1 # args.pair_reg_weight # 1
lstm_filter     = 1 # args.lstm_filter # 1
feature_type     = 1 # args.feature_type # 1
dilation_size =  1 # args.dilation_size # 1

regularize            = False
# if args.weight_regularization == 1:
#     regularize = True
#     print("Activating regularization")

include_pseudoknots = True
# if args.include_pseudoknots == 1:
#     include_pseudoknots = True
#     print("Including pseudoknots")
# else:
#     include_pseudoknots = False
#     print("Excluding pseudoknots")


expected_n_channels = 4 
# if feature_type == 0 or feature_type == -1:
#     expected_n_channels       = 4
# elif feature_type == 1:
#     expected_n_channels       = 4
# elif feature_type == 2:
#     expected_n_channels       = 4
# elif feature_type == 3:
#     expected_n_channels       = 4
# else:
#     print("Wrong ",feature_type)
#     exit(-1)
