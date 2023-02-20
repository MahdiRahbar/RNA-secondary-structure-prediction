# -*- coding: utf-8 -*-

# RNA secondary structure prediction using graph neural network and novel motif-driven analysis
# Developers: 
# License: GNU General Public License v3.0

import tensorflow as tf
from tensorflow.keras import layers

import dgl.function as fn
from dgl.nn import GATConv


"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code does not implement
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import time

import networkx as nx
import numpy as np
import tensorflow as tf
from gat import GAT
from utils import EarlyStopping

import dgl
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)


import numpy as np


"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import dgl.function as fn
import tensorflow as tf
from dgl.nn import GATConv
from tensorflow.keras import layers


class GAT(tf.keras.Model):
    def __init__(
        self,
        g,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = []
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def call(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            h = tf.reshape(h, (h.shape[0], -1))
        # output projection
        logits = tf.reduce_mean(self.gat_layers[-1](self.g, h), axis=1)
        return logits


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        model.save_weights("es_checkpoint.pb")



def accuracy(logits, labels):
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc.numpy().item()


def evaluate(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    return accuracy(logits, labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)
        g = g.to(device)

    with tf.device(device):
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        num_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        print(
            """----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d"""
            % (
                n_edges,
                n_classes,
                train_mask.numpy().sum(),
                val_mask.numpy().sum(),
                test_mask.numpy().sum(),
            )
        )

        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()
        # create model
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(
            g,
            args.num_layers,
            num_feats,
            args.num_hidden,
            n_classes,
            heads,
            tf.nn.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual,
        )
        print(model)
        if args.early_stop:
            stopper = EarlyStopping(patience=100)

        # loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False)
        loss_fcn = tf.nn.sparse_softmax_cross_entropy_with_logits

        # use optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr, epsilon=1e-8
        )

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            if epoch >= 3:
                t0 = time.time()
            # forward
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                logits = model(features, training=True)
                loss_value = tf.reduce_mean(
                    loss_fcn(
                        labels=labels[train_mask], logits=logits[train_mask]
                    )
                )
                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in model.trainable_weights:
                    loss_value = loss_value + args.weight_decay * tf.nn.l2_loss(
                        weight
                    )

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(model, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):
                        break

            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss_value.numpy().item(),
                    train_acc,
                    val_acc,
                    n_edges / np.mean(dur) / 1000,
                )
            )

        print()
        if args.early_stop:
            model.load_weights("es_checkpoint.pb")
        acc = evaluate(model, features, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GAT")
    register_data_args(parser)
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="which GPU to use. Set -1 to use CPU.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="number of hidden attention heads",
    )
    parser.add_argument(
        "--num-out-heads",
        type=int,
        default=1,
        help="number of output attention heads",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of hidden layers"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=8, help="number of hidden units"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        default=False,
        help="use residual connection",
    )
    parser.add_argument(
        "--in-drop", type=float, default=0.6, help="input feature dropout"
    )
    parser.add_argument(
        "--attn-drop", type=float, default=0.6, help="attention dropout"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="weight decay"
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop or not",
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="skip re-evaluate the validation set",
    )
    args = parser.parse_args()
    print(args)

    main(args)