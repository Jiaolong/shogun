"""
=====================
Image denoising demo
=====================
Simple binary denoising example.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from modshogun import Factor, TableFactorType, FactorGraph
from modshogun import FactorGraphObservation, FactorGraphLabels, FactorGraphFeatures
from modshogun import FactorGraphModel, GRAPH_CUT
from modshogun import GraphCut
from modshogun import StochasticSOSVM

from synthetic_grids import generate_blocks_multinomial

def define_factor_types(num_vars, len_feat, edge_table):
    """ Define factor types

        Args:
            num_vars: number of variables in factor graph
            len_feat: length of the feature vector
            edge_table: edge table defines pair-wise node indeces

        Returns:
            v_factor_types: list of all unary and pair-wise factor types
    """

    n_stats = 2 # for binary status
    v_factor_types = {}
    n_edges = edge_table.shape[0]

    # unary factors
    cards_u = np.array([n_stats], np.int32)
    w_u = np.zeros(n_stats*len_feat)
    for i in range(num_vars):
        v_factor_types[i] = TableFactorType(i, cards_u, w_u)

    # pair-wise factors
    cards_pw = np.array([n_stats, n_stats], np.int32)
    w_pw = np.zeros(n_stats*n_stats)
    for j in range(n_edges):
        v_factor_types[j + num_vars] = TableFactorType(j + num_vars, cards_pw, w_pw)

    return v_factor_types

def build_factor_graph_model(labels, feats, factor_types, edge_table, infer_alg = GRAPH_CUT):
    """ Build factor graph model

        Args:
            labels: matrix of labels [num_train_samples*len_label]
            feats: maxtrix of feats [num_train_samples*len_feat]
            factory_types: vectors of all factor types
            edge_table: matrix of pairwised edges, each row is a pair of node indeces
            infer_alg: inference algorithm (GRAPH_CUT)

        Returns:
            labels_fg: matrix of labels in factor graph format
            feats_fg: matrix of features in factor graph format
    """

    labels = labels.astype(np.int32)
    num_train_samples = labels.shape[0]
    num_vars = labels.shape[1]
    num_edges = edge_table.shape[0]
    n_stats = 2

    feats_fg = FactorGraphFeatures(num_train_samples)
    labels_fg = FactorGraphLabels(num_train_samples)

    for i in range(num_train_samples):
        cardinaities = np.array([n_stats]*num_vars, np.int32)
        fg = FactorGraph(cardinaities)

        # add unary factors
        for u in range(num_vars):
            data_u = np.array(feats[i,:], np.float64)
            inds_u = np.array([u], np.int32)
            factor_u = Factor(factor_types[u], inds_u, data_u)
            fg.add_factor(factor_u)

        # add pairwise factors
        for v in range(num_edges):
            data_p = np.array([1.0])
            inds_p = np.array(edge_table[v, :], np.int32)
            factor_p = Factor(factor_types[v + num_vars], inds_p, data_p)
            fg.add_factor(factor_p)

        # add factor graph
        feats_fg.add_sample(fg)

        # add corresponding label
        loss_weights = np.array([1.0/num_vars]*num_vars)
        fg_obs = FactorGraphObservation(labels[i,:], loss_weights)
        labels_fg.add_label(fg_obs)

    return (labels_fg, feats_fg)

def evaluation(labels_pr, labels_gt, model):
    """ Evaluation

        Args:
            labels_pr: predicted label
            labels_gt: ground truth label
            model: factor graph model

        Returns:
            ave_loss: average loss
    """

    num_train_samples = labels_pr.get_num_labels()
    acc_loss = 0.0
    ave_loss = 0.0
    for i in range(num_train_samples):
        y_pred = labels_pr.get_label(i)
        y_truth = labels_gt.get_label(i)
        acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)

    ave_loss = acc_loss / num_train_samples

    return ave_loss

def denoise_sosvm():
    import time

    # generate synthetic data
    # n_samples=20, noise=0.5, seed=None, size_x=12
    X, Y = generate_blocks_multinomial(noise=2, n_samples=20, seed=1)
    feats_train = X[:,:,:,1]
    labels_train = Y
    sz_ft = feats_train.shape
    sz_lb = labels_train.shape
    feats_train =  np.reshape(feats_train, (sz_ft[0], sz_ft[1]*sz_ft[2]))
    labels_train = np.reshape(labels_train, (sz_lb[0], sz_lb[1]*sz_lb[2]))
    len_label = labels_train.shape[1]
    len_feat = feats_train.shape[1]

    # compute full-connected edge table
    full = np.vstack([x for x in itertools.combinations(range(len_label), 2)])

    # define factor types
    factor_types = define_factor_types(len_label, len_feat, full)

    # create features and labels for factor graph mode
    (labels_fg, feats_fg) = build_factor_graph_model(labels_train, feats_train, factor_types, full, GRAPH_CUT)

    # create model and register factor types
    model = FactorGraphModel(feats_fg, labels_fg, GRAPH_CUT)

    for i in range(len(factor_types)):
        model.add_factor_type(factor_types[i])

    # Training
    # the 3rd parameter is do_weighted_averaging, by turning this on,
    # a possibly faster convergence rate may be achieved.
    # the 4th parameter controls outputs of verbose training information
    sgd = StochasticSOSVM(model, labels_fg, True, True)
    sgd.set_num_iter(150)
    sgd.set_lambda(0.0001)

    # train
    t0 = time.time()
    sgd.train()
    t1 = time.time()
    w_sgd = sgd.get_w()
    print "SGD took", t1 - t0, "seconds."

    # training error
    labels_pr = sgd.apply()
    ave_loss = evaluation(labels_pr, labels_fg, model)
    print('SGD: Average training error is %.4f' % ave_loss)

    # testing error
    # generate synthetic testing dataset

    # plot one example
    i = 10
    x, y, y_pred = X[i], Y[i], Y[i]
    y_pred = y_pred.reshape(x.shape[:2])
    fig, plots = plt.subplots(1, 3, figsize=(12, 4))
    plots[0].matshow(y)
    plots[0].set_title("ground truth")
    plots[1].matshow(np.argmax(x, axis=-1))
    plots[1].set_title("input")
    plots[2].matshow(y_pred)
    plots[2].set_title("prediction")

    for p in plots:
        p.set_xticks(())
        p.set_yticks(())

    plt.show()

if __name__=='__main__':
    print("\nDenosing demo")
    denoise_sosvm()

