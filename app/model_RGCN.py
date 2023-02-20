
# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0

"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
- https://github.com/dmlc/dgl/
"""

import argparse
import time
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import dgl
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from dgl.nn.tensorflow import RelGraphConv

class BaseRGCN(layers.Layer):
    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        num_bases,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=False,
        use_cuda=False,
    ):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = []
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def call(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

            
class EntityClassify(BaseRGCN):
    def create_features(self):
        features = tf.range(self.num_nodes)
        return features

    def build_input_layer(self):
        return RelGraphConv(
            self.num_nodes,
            self.h_dim,
            self.num_rels,
            "basis",
            self.num_bases,
            activation=tf.nn.relu,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
        )

    def build_hidden_layer(self, idx):
        return RelGraphConv(
            self.h_dim,
            self.h_dim,
            self.num_rels,
            "basis",
            self.num_bases,
            activation=tf.nn.relu,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
        )

    def build_output_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.out_dim,
            self.num_rels,
            "basis",
            self.num_bases,
            activation=partial(tf.nn.softmax, axis=1),
            self_loop=self.use_self_loop,
        )


def acc(logits, labels, mask):
    logits = tf.gather(logits, mask)
    labels = tf.gather(labels, mask)
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc


def main(args):
    # load graph data
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    else:
        raise ValueError()

    # preprocessing in cpu
    with tf.device("/cpu:0"):
        # Load from hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        train_mask = hg.nodes[category].data.pop("train_mask")
        test_mask = hg.nodes[category].data.pop("test_mask")
        train_idx = tf.squeeze(tf.where(train_mask))
        test_idx = tf.squeeze(tf.where(test_mask))
        labels = hg.nodes[category].data.pop("labels")

        # split dataset into train, validate, test
        if args.validation:
            val_idx = train_idx[: len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5 :]
        else:
            val_idx = train_idx

        # calculate norm for each edge type and store in edge
        for canonical_etype in hg.canonical_etypes:
            u, v, eid = hg.all_edges(form="all", etype=canonical_etype)
            _, inverse_index, count = tf.unique_with_counts(v)
            degrees = tf.gather(count, inverse_index)
            norm = tf.ones(eid.shape[0]) / tf.cast(degrees, tf.float32)
            norm = tf.expand_dims(norm, 1)
            hg.edges[canonical_etype].data["norm"] = norm

        # get target category id
        category_id = len(hg.ntypes)
        for i, ntype in enumerate(hg.ntypes):
            if ntype == category:
                category_id = i

        # edge type and normalization factor
        g = dgl.to_homogeneous(hg, edata=["norm"])

    # check cuda
    if args.gpu < 0:
        device = "/cpu:0"
        use_cuda = False
    else:
        device = "/gpu:{}".format(args.gpu)
        g = g.to(device)
        use_cuda = True
    num_nodes = g.number_of_nodes()
    node_ids = tf.range(num_nodes, dtype=tf.int64)
    edge_norm = g.edata["norm"]
    edge_type = tf.cast(g.edata[dgl.ETYPE], tf.int64)

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = node_tids == category_id
    target_idx = tf.squeeze(tf.where(loc))

    # since the nodes are featureless, the input feature is then the node id.
    feats = tf.range(num_nodes, dtype=tf.int64)

    with tf.device(device):
        # create model
        model = EntityClassify(
            num_nodes,
            args.n_hidden,
            num_classes,
            num_rels,
            num_bases=args.n_bases,
            num_hidden_layers=args.n_layers - 2,
            dropout=args.dropout,
            use_self_loop=args.use_self_loop,
            use_cuda=use_cuda,
        )

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        # training loop
        print("start training...")
        forward_time = []
        backward_time = []
        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )