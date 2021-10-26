#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:57:20 2021

@author: chenxiyuan

define architecture of neural network models and process of bayesian inference.
"""
from itertools import permutations
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed
from pytorch_model_summary import summary

from task import get_loader
from utils import get_pt_model


class MNISTCNN(nn.Module):
    """
    A simple CNN for MNIST classificaiton.

    Parameters
    ----------
    dropout : float, optional
        dropout rate at training phase. The default is 0.

    """
    def __init__(self, dropout: float = 0):
        """
        A simple CNN for MNIST classificaiton.

        Parameters
        ----------
        dropout : float, optional
            dropout rate at training phase. The default is 0.

        """
        super(MNISTCNN, self).__init__()
        self.dropout = dropout
        self.input_shape = (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.out = nn.Linear(32 * 5 * 5, 10)    
        
    def forward(self, x):  
        dropout = nn.Dropout(self.dropout)
        x = F.relu(self.conv1(x))
        x = dropout(x)
        x = self.maxpool1(x)
        x = dropout(x)
        x = F.relu(self.conv2(x))
        x = dropout(x)
        x = self.maxpool2(x)
        x = x.reshape([x.shape[0], -1])
        x = dropout(x)
        output = self.out(x)
        return output


def load_model(dataset: str = 'mnist',
               architecture: str = 'cnn',
               index: int = 0, 
               dropout_train: float = 0, 
               dropout_test: float = 0,
               device: str = 'cuda',
               multigpu: bool = True):
    """
    Load a well-trained model and change its dropout rate.

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
    device : str, optional
        device on which the model is loaded. The default is 'cuda'.
    multigpu : bool, optional
        whether perform Data Parallelism on multiplt GPUs. The default is True.

    Returns
    -------
    model : torch.nn.Module
        pytorch neural network classifier.    
    """
    # define model
    if dataset == 'mnist' and architecture == 'cnn':
        model = MNISTCNN(dropout_test)
    
    model = model.to(device)
    # load pre-trained weights
    pt = get_pt_model(dataset, architecture, index, dropout_train)
    model.load_state_dict(torch.load(pt))
    model.eval()
    # perform Data Parallelism
    if multigpu:
        model = nn.DataParallel(model)
    return model


def get_combination_all(n_choice: int, n_channel: int):
    """
    Return all possible discretized neural network outputs.

    Parameters
    ----------
    n_choice : int
        number of labels.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    combination_all : np.ndarray, shape as [n_combination, n_channel]
        all possible discretized neural network outputs.

    """
    # up to 256 labels when using uint8
    combination_all = list(permutations(range(n_choice), n_channel))
    combination_all = np.array(combination_all, dtype='uint8')
    return combination_all


def output2index(output: np.ndarray, n_channel: int):
    """
    Discretize the neural network output by sorting the first *n_channel* max output 
    channels and return the index of this discretized output in all possible combination.
    
    For example:
        for a 10-label classification task and n_channel=3,

        for an output of `[0, 0.7, 0, 0, 0.1, 0, 0.2, 0, 0, 0]`, the first 3 maximum
        channels are `[1, 6, 4]`
    
        all possible combinations of 3 channels are `[0, 1, 2]`, `[0, 1, 3]`, `[0, 1, 4]`, ...
        
        `[1, 6, 4]` is the *155th* of all possible combinations
        
        hence `[0, 0.7, 0, 0, 0.1, 0, 0.2, 0, 0, 0]` will be converted as 155
    

    Parameters
    ----------
    output : np.ndarray, shape as [n_sample, n_choice]
        neural network classifier outputs.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    idx : TYPE
        DESCRIPTION.

    """
    # [n_combination_possible, n_channel]
    combination_all = get_combination_all(output.shape[-1], n_channel)
    combination = np.flip(np.argsort(output, axis=-1)[..., -n_channel:], axis=-1)
    n_sample, n_channel = combination.shape
    # [n_sample, n_combination, n_channel]
    combination_repeat = np.stack([combination_all]*n_sample, axis=0)
    idx = combination_repeat == combination[:, np.newaxis]
    idx = np.argmax(np.sum(idx, axis=-1), axis=-1)
    return idx


def get_likelihood(output_of_category: list, n_channel: int):
    """
    Estimate the likelihood of discretized neural network outputs in each class.

    Parameters
    ----------
    output_of_category : list, shape as [n_choice] [n_output, n_choice]
        list of arrays of neural network output of each category.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    likelihood : np.ndarray, shape as [n_combination_possible, n_choice]
        likelihood of discretized neural network outputs in each class.

    """
    n_choice = output_of_category[0].shape[-1]
    combination_all = get_combination_all(n_choice, n_channel)
    n_combination = len(combination_all)

    likelihood = np.zeros([n_combination, n_choice])    
    for choice, i in enumerate(output_of_category):
        n_sample, n_choice = i.shape
        index = output2index(i, n_channel)
        # counting combination of outputs of all choices
        combination_i, count_i = np.unique(index, False, False, True)
        likelihood_i = count_i / np.sum(n_sample)
        likelihood[combination_i, choice] = likelihood_i
        # assign unshown combination with minimum likelihood
        idx_unshown = np.delete(np.arange(n_combination), index)
        likelihood[idx_unshown, choice] = min(likelihood_i)

    
    return likelihood


def bayesian_inference(trial: np.ndarray, 
                       boundary: float, 
                       likelihood: np.ndarray,
                       n_channel: int):
    """
    Perform bayesian inference with neural network predictions as observations.

    Parameters
    ----------
    trial : np.ndarray, shape as [n_trial, len_trial, n_channel]
        trials of neural network outputs.
    boundary : float
        decision made when any posterior belief surpass this boundary.
    likelihood : np.ndarray, shape as [n_combination_possible, n_choice]
        likelihood of discretized neural network outputs in each class.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    belief_post : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all posterior beliefs through the process of bayesian inference.
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by bayesian inference.

    """

    n_trial, len_trial, n_choice = trial.shape
    idx_combination = [output2index(trial[:, i]) for i in range(len_trial)]
    idx_combination = np.stack(idx_combination, axis=1)

    belief_post = np.full([n_trial, len_trial, n_choice], 1 / n_choice)
    for t in range(1, len_trial):
        # likelihood.shape = [n_choice, n_combination]
        # p(x_t|A_i) for all i(labels), shape as [n_trial, n_choice]
        prob_xt = likelihood[idx_combination[:, t]]
        # p(x_t|A_i)*p(A_i|X_1:t-1)
        prob_product = prob_xt * belief_post[:, t-1]
        belief_post[:, t] = prob_product / np.sum(prob_product, -1, keepdims=True)
    
    # decision-making based on posteror belief
    # [n_trial, n_timestep]
    evidence_max = np.max(belief_post, axis=-1)
    channel_max = np.argmax(belief_post, axis=-1)
    hit_boundary = evidence_max >= boundary
    # force decision based on last evidence if never hit boundary
    never_hit = np.logical_not(np.sum(hit_boundary, -1))
    hit_boundary[never_hit, -1] = True
    # [n_trial]
    response_time = np.argmax(hit_boundary, axis=-1)
    choice = np.max((channel_max + 1) * hit_boundary, axis=-1) - 1
    
    return belief_post, response_time, choice


if __name__ == '__main__':
    model = load_model()
    pass