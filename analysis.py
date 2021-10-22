#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:24:26 2021

@author: chenxiyuan

plot figures for result analysis
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed

from task import get_output_of_category, load_acc, load_bayes, load_responsetime
from traintest import PROPORTION, DROPOUT, METHOD, METHOD_TITLE, EPSILON, EPSILON_FIGURE
from utils import get_dir


def plot_heatmap(dataset: str = 'mnist', 
                 architecture: str = 'cnn',
                 index: int = 0, 
                 dropout: list = DROPOUT, 
                 method: list = METHOD, 
                 epsilon: list = EPSILON, 
                 subset: str = 'test', 
                 proportion: float = PROPORTION, 
                 repeat: int = 100,
                 len_trial: int = 25, 
                 n_trial: int = 10, 
                 boundary: float = 0.99, 
                 n_channel: int = 3,
                 mode: str = 'attack'):
    """
    Plot accuracy heatmap of all dropout rates and attacks combinations.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout : list, optional
        training and testing dropout rates. The default is DROPOUT.
    method : list, optional
        attack methods. The default is METHOD.
    epsilon : list, optional
        epsilons of each attack. The default is EPSILON.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.
    boundary : float, optional
        posterior belief decision boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.
    mode : str, optional
        DESCRIPTION. The default is 'attack'.

    Returns
    -------
    acc : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.

    """
    subset = 'test'
    n_dropout = len(dropout)
    n_subplot = len(method)
    acc = np.zeros([n_dropout, n_dropout, n_subplot])

    idx_ijk = []    
    args = []
    for idx_i, i in enumerate(dropout):
        for idx_j, j in enumerate(dropout):
            for idx_k, (m, e) in enumerate(zip(method, epsilon)):
                idx_ijk.append((idx_i, idx_j, idx_k))
                # args for loading acc before bayes
                arg = [mode, dataset, architecture, index, i, j, m, e, subset, proportion]
                if mode == 'bayes':
                    arg += [repeat, len_trial, n_trial, boundary, n_channel]
                
                args.append(arg)
    result = Parallel(-1)(delayed(load_acc)(*i) for i in args)

    for (idx_i, idx_j, idx_k), r in zip(idx_ijk, result):
        acc[idx_i, idx_j, idx_k] = r

    # 4 is for acc(0, 0), max(acc), max(acc).train_level, max(acc).test_level
    table = np.zeros([4, n_subplot])
    table[0] = acc[0, 0, :]
    table[1] = np.max(acc, axis=(0, 1))

    fig, axes = plt.subplots(1, n_subplot, figsize=(n_subplot*3, 3), sharey=True)
    xticklabels = ['%.1f'%i for i in dropout]
    yticklabels = ['%.1f'%i for i in dropout]
    plt.setp(axes, xticks=range(n_dropout), xticklabels=xticklabels,
             yticks=range(n_dropout), yticklabels=yticklabels)
    subtitle = METHOD_TITLE
    for i in range(n_subplot):
        a = acc[:, :, i]
        im = axes[i].imshow(a, norm=Normalize(0, 1))
        idx = np.unravel_index(np.argmax(a), a.shape)
        table[2, i] = dropout[idx[0]]
        table[3, i] = dropout[idx[1]]
        t = ' (%.1f, %.1f: %.4f)'%(table[2, i], table[3, i], table[1, i])
        axes[i].set_title(subtitle[i] + t)

    axes[0].set_ylabel('Training Dropout Rate')
    axes[n_subplot//2].set_xlabel('Testing Dropout Rate')
    # color bar for heatmap
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.suptitle(mode)
    fig.show()

    return acc, table


def plot_output_histogram(dataset: str = 'mnist', 
                          architecture: str = 'cnn',
                          index: int = 0, 
                          dropout_train: float = 0, 
                          dropout_test: float = 0, 
                          method: str = 'clean', 
                          epsilon: float = 0, 
                          subset: str = 'test', 
                          proportion: float = PROPORTION, 
                          repeat: int = 100,
                          bins: int = 20):
    """
    Plot histogram of outputs of each channel in each class.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_train : float, optional
        dropout rate at training phase. The default is 0.
    dropout_test : float, optional
        dropout rate at test phase. The default is 0.
    method : str, optional
        adversarial attack method. The default is 'clean'.
    epsilon : float, optional
        perturbation threshold of attacks. The default is 0.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100.
    bins : int, optional
        number of bins of histogram. The default is 20.

    Returns
    -------
    None.

    """
    output_of_category = get_output_of_category(dataset, architecture, index, dropout_train,
                                                dropout_test, method, epsilon, 
                                                subset, repeat)
    n_choice = len(output_of_category)
    fig, axes = plt.subplots(n_choice, n_choice, True, True,
                             figsize=(n_choice*3, n_choice*3))
    title = [dataset, architecture, dropout_train, dropout_test, method, epsilon]
    title = ' '.join([str(i) for i in title])
    fig.suptitle(title, fontsize=80)
    for i in range(n_choice):
        for j in range(n_choice):
            axes[i][j].hist(output_of_category[i][:, j], bins)
            axes[i][j].set_title('image %d - output %d'%(i, j))
    
    fig.show()
    d = get_dir(dataset, 'figure')
    plt.savefig(d+title+'.svg')


def plot_rt_histogram(dataset: str = 'mnist', 
                      architecture: str = 'cnn',
                      index: int = 0, 
                      dropout_train: float = 0, 
                      dropout_test: float = 0, 
                      method: str = 'clean', 
                      epsilon: float = 0, 
                      subset: str = 'test', 
                      proportion: float = PROPORTION, 
                      repeat: int = 100,
                      len_trial: int = 25, 
                      n_trial: int = 10, 
                      boundary: float = 0.99, 
                      n_channel: int = 3,):
    """
    Plot histogram of response time of bayesian inference.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_train : float, optional
        dropout rate at training phase. The default is 0.
    dropout_test : float, optional
        dropout rate at test phase. The default is 0.
    method : str, optional
        adversarial attack method. The default is 'clean'.
    epsilon : float, optional
        perturbation threshold of attacks. The default is 0.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.
    boundary : float, optional
        posterior belief decision boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.

    Returns
    -------
    None.

    """

    response_time = load_bayes(dataset, architecture, index, dropout_train, dropout_test, 
                               method, epsilon, subset, repeat, len_trial, n_trial, 
                               boundary, n_channel)
    fig = plt.figure()
    plt.hist(response_time)
    title = [dataset, architecture, dropout_train, dropout_test, method, epsilon]
    title = ' '.join([str(i) for i in title])
    plt.title(title)
    plt.show
    

def plot_epsilon_accrt(dataset: str = 'mnist', 
                       architecture: str = 'cnn',
                       index: int = 0, 
                       dropout_train: float = 0, 
                       dropout_test: float = 0, 
                       method: str = 'pgd', 
                       epsilon: list = EPSILON_FIGURE, 
                       subset: str = 'test', 
                       proportion: float = PROPORTION, 
                       repeat: int = 100,
                       len_trial: int = 25, 
                       n_trial: int = 10, 
                       boundary: float = 0.99, 
                       n_channel: int = 3,):
    """
    Plot relationship of epsilon-accuracy and epsilon-response_time.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_train : float, optional
        dropout rate at training phase. The default is 0.
    dropout_test : float, optional
        dropout rate at test phase. The default is 0.
    method : str, optional
        adversarial attack method. The default is 'pgd'.
    epsilon : list, optional
        perturbation thresholds of attack. The default is EPSILON_FIGURE.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.
    boundary : float, optional
        posterior belief decision boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.

    Returns
    -------
    None.

    """
    fig = plt.figure()
    y_left = fig.add_subplot(111)
    y_left.set_xlabel('ϵ')
    y_left.set_ylabel("Accuracy")
    y_left.set_ylim(0, 1)
    
    y_right = y_left.twinx()
    y_right.set_ylabel("Response Time")
    # y_right.set_ylim(0, 18)
    
    n_epsilon = len(epsilon)
    # dropout_00, dropout_optimal, bayes
    n_accuracy = 3
    # bayesian
    accuracy = np.zeros([n_epsilon, n_accuracy])
    response_time = np.zeros([n_epsilon])
    for i, eps in enumerate(epsilon):
        # load acc of dropout_00 and dropout_optimal
        for j, (dtrain, dtest) in enumerate([(0, 0), (dropout_train, dropout_test)]):
            args = ['attack', dataset, architecture, index, dtrain, dtest, method, 
                    eps, subset, proportion]
            accuracy[i, j] = load_acc(*args)
        
        # load acc and rt of bayes
        args = ['bayes', dataset, architecture, index, dropout_train, dropout_test, 
                method, eps, subset, proportion, repeat, len_trial, n_trial, boundary, 
                n_channel]
        accuracy[i, 2] = load_acc(*args)
        response_time[i] = np.mean(load_responsetime(*args[1:]))

    label_accuracy = ['Naive Acc', 'Dropout Acc', 'Bayes Acc']
    color = ['red', 'orange', 'green']
    for i in range(n_accuracy):
        y_left.plot(epsilon, accuracy[:, i], label=label_accuracy[i], color=color[i])
    # fake line to insert the legend
    y_left.plot([], [], label='Bayes RT', ls='--', color=color[-1])
    y_right.plot(epsilon, response_time, ls='--', color=color[-1])
    y_left.legend()
    title = [dataset, architecture, index, dropout_train, dropout_test, method]
    title = 'ϵ-Acc/RT ' + ' '.join([str(i) for i in title])
    plt.title(title)
    plt.show()
        
        


if __name__ == '__main__':
    acc0, table0 = plot_heatmap()
    acc1, table1 = plot_heatmap(mode='bayes')
    plot_epsilon_accrt(dropout_train=0.2, dropout_test=0.6, proportion=0.1)
    pass