from tensorflow.keras.utils import *
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from tensorflow.python.keras import layers, optimizers

from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.layers import Input, Dense, Conv1D, Convolution2D, Activation
from tensorflow.keras.layers import add, Add, Dropout, BatchNormalization, Reshape, Lambda
from tensorflow.keras.layers import Bidirectional, Multiply,concatenate, Embedding, ConvLSTM2D

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2



def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def rna_pair_prediction_bin_gcn5(node_num = None, node_dim=4, hidden_dim=100, voc_edges_in = 2, voc_edges_out = 1, voc_nodes_out = 2, num_gcn_layers = 10, num_lstm_layers = 2, lstm_filter = 8, aggregation = "mean", regularize=False, dropout_rate = 0.25, dilation_size = 1, filter_size=3, feature_type=0):
    dropout_value = dropout_rate
    # Node embedding
    node_input = Input(shape = (node_num,node_dim))
    #nodes_embedding = Dense(hidden_dim, input_dim=(node_num, node_dim),use_bias=False)(node_input) # B x V x H
    nodes_embedding = InstanceNormalization()(node_input)
    if regularize:
      nodes_embedding = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = dilation_size, kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(nodes_embedding)
      nodes_embedding = Activation('relu')(nodes_embedding)
      nodes_embedding = BatchNormalization()(nodes_embedding)
      nodes_embedding = Dropout(dropout_value)(nodes_embedding)
    else:
      nodes_embedding = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = dilation_size, kernel_initializer = 'he_normal', padding = 'same')(nodes_embedding)
      nodes_embedding = Activation('relu')(nodes_embedding)
      nodes_embedding = BatchNormalization()(nodes_embedding)
      nodes_embedding = Dropout(dropout_value)(nodes_embedding)
    
       
    # Edge weight embedding: Input edge distance matrix (batch_size, num_nodes, num_nodes)
    if feature_type == 0 or feature_type == 1 or feature_type == -1:
        edges_value_input = Input(shape = (node_num,node_num,1))
        edges_value_embedding = InstanceNormalization()(edges_value_input)
        #edges_value_embedding = Dense(hidden_dim//2, input_dim=(node_num,node_num,1),use_bias=False)(edges_value_input) # B x V x V x H
        if regularize:
          edges_value_embedding = Convolution2D(hidden_dim//2, kernel_size = (filter_size, filter_size), dilation_rate = (dilation_size,dilation_size), kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(edges_value_embedding)
        else:
          edges_value_embedding = Convolution2D(hidden_dim//2, kernel_size = (filter_size, filter_size), dilation_rate = (dilation_size,dilation_size), kernel_initializer = 'he_normal', padding = 'same')(edges_value_embedding)
        edges_value_embedding = Activation('relu')(edges_value_embedding)
        edges_value_embedding = BatchNormalization()(edges_value_embedding)
        edges_value_embedding = Dropout(dropout_value)(edges_value_embedding)
    elif feature_type == 2 or feature_type == 3:
        edges_value_input = Input(shape = (node_num,node_num,22))
        edges_value_embedding = InstanceNormalization()(edges_value_input)
        #edges_value_embedding = BatchNormalization()(edges_value_input)
        #edges_value_embedding = Dense(hidden_dim//2, input_dim=(node_num,node_num,22),use_bias=False)(edges_value_embedding) # B x V x V x H
        if regularize:
          edges_value_embedding = Convolution2D(hidden_dim//2, kernel_size = (filter_size, filter_size), dilation_rate = (dilation_size,dilation_size), kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(edges_value_embedding)
        else:
          edges_value_embedding = Convolution2D(hidden_dim//2, kernel_size = (filter_size, filter_size), dilation_rate = (dilation_size,dilation_size), kernel_initializer = 'he_normal', padding = 'same')(edges_value_embedding)
        edges_value_embedding = Activation('relu')(edges_value_embedding)
        edges_value_embedding = BatchNormalization()(edges_value_embedding)
        edges_value_embedding = Dropout(dropout_value)(edges_value_embedding)
      
    
    # Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
    edges_adj_input = Input(shape = (node_num,node_num))
    edges_adj_embedding = Embedding(voc_edges_in, hidden_dim//2)(edges_adj_input) # B x V x V x H
    
    # merge edge embedding
    edge_merge_embedding = concatenate([edges_value_embedding, edges_adj_embedding])
    
   
    
    
    
    
    
    
    
    
    
    ## ------------ 
    np.fill_diagonal(b[0].squeeze(), 1) 

    src, dst = np.nonzero(b[0].squeeze())

    edge_feat = np.zeros(src.shape)
    # edge_dst = np.zeros(dst.shape)

    for i, (j, k) in enumerate(zip(src, dst)):
        edge_feat[i,] = c[0,j,k]

    g = dgl.graph((src, dst))
    g.ndata['feat'] = tf.convert_to_tensor(a[0].squeeze())
    g.edata['feat'] = tf.convert_to_tensor(edge_feat.squeeze())
    #####-------
    
    ################ (1) Define ResidualGatedGCNLayer ################
    d_rate = dilation_size
    for layer in range(num_gcn_layers):
        x_in,e_in = nodes_embedding, edge_merge_embedding
        
        #class ResidualGatedGCNLayer(nn.Module):
        """Convnet layer with gating and residual connection.
        """
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
    
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
    
        ################# (1.1) Edge convolution (class EdgeFeatures(nn.Module))
        """Convnet features for edges.
        e_ij = U*e_ij + V*(x_i + x_j)
        """
        #edge_Ue = Dense(hidden_dim,use_bias=True)(e_in) # B x V x V x H
        if regularize:
          edge_Ue = Convolution2D(hidden_dim, kernel_size = (filter_size, filter_size), dilation_rate = (d_rate,d_rate), kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(e_in)
        else:
          edge_Ue = Convolution2D(hidden_dim, kernel_size = (filter_size, filter_size), dilation_rate = (d_rate,d_rate), kernel_initializer = 'he_normal', padding = 'same')(e_in)
        edge_Ue = Activation('relu')(edge_Ue)
        edge_Ue = BatchNormalization()(edge_Ue)
        edge_Ue = Dropout(dropout_value)(edge_Ue)
        
        
        #edge_Vx = Dense(hidden_dim,use_bias=True)(x_in) # B x V x H
        edge_Vx = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        edge_Vx = Activation('relu')(edge_Vx)
        edge_Vx = BatchNormalization()(edge_Vx)
        edge_Vx = Dropout(dropout_value)(edge_Vx)
        
        edge_Wx = K.expand_dims(edge_Vx, 2) # Extend Vx from "B x V x H" to "B x V x 1 x H"
        edge_Vx = K.expand_dims(edge_Vx, 1) # extend Vx from "B x V x H" to "B x 1 x V x H"
        '''
        e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        '''
        #e_new = Ue + Vx + Wx
        edge_convnet = Add()([edge_Ue, edge_Vx, edge_Wx]) # B x V x V x H
        #####################################################################
    
        ################# (1.2) Compute edge gates ######################### 
        edge_convnet_gate = Activation('sigmoid')(edge_convnet) # B x V x V x H
        #####################################################################
    
        ################# (1.3) Node convolution (class NodeFeatures(nn.Module))
        # self.node_feat = NodeFeatures(hidden_dim, aggregation)
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)
    
        Returns:
            node_convnet: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
    
        """Convnet features for nodes.
    
        Using `sum` aggregation:
            x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
        Using `mean` aggregation:
            x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
        """
    
        #node_Ux = Dense(hidden_dim,use_bias=True)(x_in)  # B x V x H
        node_Ux = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        node_Ux = BatchNormalization()(node_Ux)
        node_Ux = Activation('relu')(node_Ux)
        node_Ux = Dropout(dropout_value)(node_Ux)
        
        #node_Vx = Dense(hidden_dim,use_bias=True)(x_in)  # B x V x H
        node_Vx = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        node_Vx = BatchNormalization()(node_Vx)
        node_Vx = Activation('relu')(node_Vx)
        node_Vx = Dropout(dropout_value)(node_Vx)
        
        node_Vx = K.expand_dims(node_Vx, 1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        node_gateVx = Multiply()([edge_convnet_gate, node_Vx])  # B x V x V x H
        if aggregation=="mean":
            ReduceSum = Lambda(lambda z: K.sum(z, axis=2))
            node_gateVx_sum = ReduceSum(node_gateVx)
            edge_convnet_gate_sum = ReduceSum(edge_convnet_gate)
            
            divResult = Lambda(lambda x: x[0]/(x[1]+1e-20))
            mean_node = divResult([node_gateVx_sum,edge_convnet_gate_sum])
            node_convnet = Add()([node_Ux, mean_node])  # B x V x H
        elif aggregation=="sum":
            ReduceSum = Lambda(lambda z: K.sum(z, axis=2))
            node_gateVx_sum = ReduceSum(node_gateVx)
            node_convnet = Add()([node_Ux, node_gateVx_sum])  # B x V x H
    
        ################# (1.4) Batch normalization for edge and node
        """Batch normalization for edge features.
        """
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
    
        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
    
        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        # input: edge_convnet
        edge_convnet = BatchNormalization()(edge_convnet)
        node_convnet = BatchNormalization()(node_convnet)
    
        ################# (1.5) Relu Activation for edge and node
        edge_convnet = Activation('relu')(edge_convnet)
        node_convnet = Activation('relu')(node_convnet)
    
        ################# (1.6) Residual connection
        node_out = Add()([x_in, node_convnet])
        edge_out = Add()([e_in, edge_convnet])
    
        ################# (1.7) Update embedding
        nodes_embedding = node_out  # B x V x H
        edge_merge_embedding = edge_out  # B x V x V x H
        
        d_rate = d_rate*2
        if d_rate > 4:
            d_rate = 4
    
    
    ################ (2) Define MLP classifiers for edge ################
    #self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
    
    #for layer in range(num_mlp_layers):
    #    edge_merge_embedding = Dense(hidden_dim,use_bias=True)(edge_merge_embedding) # B x V x V x H
    #    edge_merge_embedding = Activation('relu')(edge_merge_embedding)
    #y_pred_edges = Dense(voc_edges_out,use_bias=True)(edge_merge_embedding) # B x V x V x voc_edges_out for probability
    


    # ##### - -- -- --- - --  - -- -- --  - - - -- - - -- - -- - -- - - 
    # ### The following lines are commented out intentionally so that  
    # ##### the lstm part is excluded from the framework
    # ##### - -- -- --- - --  - -- -- --  - - - -- - - -- - -- - -- - - 
    # for i in range(num_lstm_layers):
    #     # LSTM
    #     tower_shape = K.int_shape(edge_merge_embedding)
    #     #print('1: ',tower_shape,' ',tower_shape[1],' ',tower_shape[2])
    #     edge_merge_embedding = Lambda(ReshapeConv_to_LSTM)(edge_merge_embedding)
    #     tower_shape = K.int_shape(edge_merge_embedding)
    #     #print('2: ',tower_shape,' ',tower_shape[1],' ',tower_shape[2])
    #     edge_merge_embedding = Bidirectional(ConvLSTM2D(filters=lstm_filter, kernel_size=(filter_size, filter_size),
    #                     input_shape=(None, tower_shape[1],tower_shape[2],tower_shape[-1]),
    #                     padding='same', return_sequences=True,  stateful = False), merge_mode='concat')(edge_merge_embedding)
    #     tower_shape = K.int_shape(edge_merge_embedding)
    #     #print('3: ',tower_shape,' ',tower_shape[1],' ',tower_shape[2])
    #     LSTM_to_conv_dims = (tower_shape[1],tower_shape[2],tower_shape[-1])
    #     edge_merge_embedding = Lambda(ReshapeLSTM_to_Conv)(edge_merge_embedding)
        
    #     #tower_shape = K.int_shape(tower)
    #     #print('4: ',tower_shape,' ',tower_shape[1],' ',tower_shape[2])
    


    edge_merge_embedding = Activation('relu')(edge_merge_embedding)
    edge_merge_embedding = BatchNormalization()(edge_merge_embedding)
    edge_merge_embedding = Dropout(dropout_value)(edge_merge_embedding)
    edge_merge_embedding = Convolution2D(1, kernel_size = (filter_size, filter_size), dilation_rate=(d_rate, d_rate), padding = 'same')(edge_merge_embedding)
    
    nt_tower = Activation('sigmoid', name = "nt_out")(edge_merge_embedding)
    pair_tower2 = Activation('sigmoid', name = "pair_out2")(edge_merge_embedding) # subregion cross entropy

    ################ (3) Define MLP classifiers for node ################
    #self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)
    
    #for layer in range(num_mlp_layers):
    #    nodes_embedding = Dense(hidden_dim,use_bias=True)(nodes_embedding) # B x V x H
    #    nodes_embedding = Activation('relu')(nodes_embedding)
    #y_pred_nodes = Dense(voc_nodes_out,use_bias=True)(nodes_embedding) # B x V x voc_nodes_out for probability
        
    
    edge_pred_model = Model([node_input, edges_value_input,edges_adj_input], [pair_tower2,nt_tower])   
    #node_pred_model = Model([node_input, edges_value_input,edges_adj_input], y_pred_nodes)  
    return edge_pred_model
