#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:57:20 2021

@author: chenxiyuan

define architecture of neural network models and process of bayesian inference.
"""
from itertools import permutations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from utils import get_pt_model
from resnet import resnet20


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNISTCNN(nn.Module):
    """
    A simple CNN for MNIST classificaiton. This model refers to the one used for 
    adversarial training in (Madry et.al., 2017, https://arxiv.org/abs/1706.06083).
    the offcial TF implementation is on https://github.com/MadryLab/mnist_challenge
    the pytorch implemantation refers to https://github.com/ylsung/pytorch-adversarial-training

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
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)  
        self.fc2 = nn.Linear(1024, 10)  
        
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
        x = self.fc1(x)
        x = dropout(x)
        output = self.fc2(x)
        return output
        

class LSTM(nn.Module):
    """
    LSTM for IMDB classification task.
    structure and hyperparameters of the LSTM follow http://arxiv.org/abs/1509.01626
    the merged matrix multiplication follows https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

    Parameters
    ----------
    input_size : int, optional
        input vector dims. The default is 300.
    hidden_size : int, optional
        hidden state dims. The default is 512.
    proj_size : int, optional
        number of labels. The default is 2.
    dropout : float, optional
        dropout rate. The default is 0.
    return_sequence : bool, optional
        whether return predictions of each token. The default is False.

    """
    def __init__(self, 
                 input_size: int = 300, 
                 hidden_size: int = 512,
                 dropout: float = 0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        
        self.init_weights()
        pass

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, init_states: torch.Tensor = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        dropout = nn.Dropout(self.dropout)
        batch_size, len_seq, _ = x.size()
        sequence = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        for t in range(len_seq):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            gates = dropout(gates)
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :self.hidden_size]), # input
                torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]), # forget
                torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),
                torch.sigmoid(gates[:, self.hidden_size*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            sequence.append(h_t.unsqueeze(0))
        
        sequence = torch.cat(sequence, dim=0)
        # reshape from shape (sequence, batch, hidden_size | proj_size) to 
        # (batch, sequence, hidden_size | proj_size)
        sequence = sequence.transpose(0, 1).contiguous()  
        return sequence          



class IMDBLSTM(nn.Module):
    def __init__(self,
                 input_size: int = 300, 
                 hidden_size: int = 512,
                 proj_size: int = 2,
                 dropout: float = 0):
        self.lstm = LSTM(input_size, hidden_size, dropout)
        self.project_layer = nn.Linear(hidden_size, proj_size) 
    def forward(self, x: torch.Tensor, init_states: torch.Tensor = None):
        batch_size, len_seq, _ = x.size()
        sequence = self.lstm(x)
        output = torch.mean(sequence, dim=1)
        output = self.project_layer(output)
        
        return output


class CIFAR10RESNET(nn.Module):
    """
    Resnet20 for CIFAR10 classificaiton task. Unlike other networks, dropout
    is not suitable for Resnet because of the batch-norm, so random noise 
    is introduced by randomly downsampling images before feeding them to 
    Resnet.

    Parameters
    ----------
    dropout : float, optional
        percentage of pixels to discard during downsample. The default is 0.
    pretrained_th : str, optional
        `.th` file path that contains pretrained model weights. Should be 
        downloaded from https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th
        When given None, model will be trained from scratch.
        The default is None.
    """
    def __init__(self, dropout: float = 0, pretrained_th: str = None):

        super(CIFAR10RESNET, self).__init__()
        self.dropout = dropout
        self.downsample_size = int(32 * (1-dropout) ** 0.5)
        self.resnet = resnet20()
        if pretrained_th is not None:
            state_dict = torch.load(pretrained_th)['state_dict']
            for i in list(state_dict.keys()):
                # "module.conv1.weight" --> "conv1.weight"
                state_dict[i[7:]] = state_dict[i]
                state_dict.pop(i)
            self.resnet.load_state_dict(state_dict)

        
    def get_random_index(self):
        idx = torch.randperm(32)[:self.downsample_size]
        idx = torch.sort(idx)[0]
        return idx


    def downsample(self, x: torch.Tensor):
        """
        Randomly downsample image.

        Parameters
        ----------
        x : torch.Tensor
            cifar-10 image.

        Returns
        -------
        x_downsampled : torch.Tensor
            downsampled image.

        """
        x_downsampled = []
        idx_row = self.get_random_index()
        for i in idx_row:
            idx_column = self.get_random_index()
            x_downsampled.append(x[:, :, i, idx_column])
        x_downsampled = torch.stack(x_downsampled, 2)
        
        return x_downsampled
    
    def forward(self, x: torch.Tensor):
        x = self.downsample(x)
        y = self.resnet(x)
        return y
    

class SPEECHCOMMANDSDEEPSPEECH(nn.Module):
    def __init__(self, dropout: float = 0, return_sequence: bool = False):
        super(SPEECHCOMMANDSDEEPSPEECH, self).__init__()
        self.dropout = dropout
        self.spectrogram = MelSpectrogram()
        self.conv1 = nn.Conv1d(128, 128, 3, 1)
        # self.conv2 = nn.Conv1d(128, 128, 3, 1)
        # self.conv3 = nn.Conv2d(32, 32, (21, 11), (2,1))
        self.lstm1 = LSTM(128, 128, dropout)
        self.lstm2 = LSTM(128, 128, dropout)
        # self.lstm3 = LSTM(128, 128, dropout)
        # self.lstm4 = LSTM(128, 128, dropout)
        self.fc1 = nn.Linear(128, 128) 
        self.project_layer = nn.Linear(128, 35) 


    def forward(self, x: torch.Tensor, init_states: torch.Tensor = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        dropout = nn.Dropout(self.dropout)
        batch_size = x.shape[0]
        
        # [batch_size, 128, 81]
        x = self.spectrogram(x)

        x = self.conv1(x)
        x = dropout(x)

        x = torch.permute(x, [0, 2, 1])
        
        x = self.lstm1(x)
        x = self.lstm2(x)
        
        x = self.fc1(x)
        x = torch.mean(x, dim=1)
        output = self.project_layer(x)
        
        return output
        
        return output         


def load_model(dataset: str = 'mnist',
               architecture: str = 'cnn',
               index: int = 0, 
               dropout_train: float = 0, 
               dropout_test: float = 0,
               device: torch.device = DEVICE,
               multigpu: bool = False,
               return_untrained: bool = False):
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
    return_sequence : bool, optional
        whether RNN returns predictions of each token. The default is False.

    Returns
    -------
    model : torch.nn.Module
        pytorch neural network classifier.    
    """
    if return_untrained:
        dropout_test = dropout_train
        if dataset == 'cifar10' and architecture == 'resnet':
            # https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th
            th = 'data/cifar10/others/resnet20-12fca82f.th'
        else:
            th = None

    # define model
    if dataset == 'mnist' and architecture == 'cnn':
        model = MNISTCNN(dropout_test)
    elif dataset == 'imdb' and architecture == 'lstm':
        model = IMDBLSTM(dropout=dropout_test)
    elif dataset == 'cifar10' and architecture == 'resnet':
        model = CIFAR10RESNET(dropout_test, th)
    elif dataset == 'speechcommands' and architecture == 'deepspeech':
        model = SPEECHCOMMANDSDEEPSPEECH(dropout_test)
    else:
        raise Exception('Wrong dataset and architecture combination: %s + %s'%(dataset, architecture))
    
    if return_untrained:
        model = model.to(device)
        return model

    # load pre-trained weights
    pt = get_pt_model(dataset, architecture, index, dropout_train)
    model.load_state_dict(torch.load(pt))
    model = model.to(device)
    model.eval()
    # perform Data Parallelism
    if device == 'cuda' and multigpu:
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

def threshold_decision_process(evidence: np.ndarray, boundary: float):
    """
    Make dicision when accumulated evience reaches threshold/bounday. 
    For bayesian inference, the evidence will be posterior beliefs.

    Parameters
    ----------
    evidence : np.ndarray
        DESCRIPTION.
    boundary : float
        DESCRIPTION.

    Returns
    -------
    response_time : np.ndarray
        number of timesteps needed to make the decision.
    choice : np.ndarray
        final decisions.

    """
    # decision-making based on posteror belief
    # [n_trial, n_timestep]
    evidence_max = np.max(evidence, axis=-1)
    channel_max = np.argmax(evidence, axis=-1)
    hit_boundary = evidence_max >= boundary
    # force decision based on last evidence if never hit boundary
    never_hit = np.logical_not(np.sum(hit_boundary, -1))
    hit_boundary[never_hit, -1] = True
    # [n_trial]
    response_time = np.argmax(hit_boundary, axis=-1)
    choice = np.max((channel_max + 1) * hit_boundary, axis=-1) - 1
    
    return response_time, choice


def cumsum_inference(trial: np.ndarray, boundary: float):
    """
    Perform cumsum inference with neural network predictions as observations.

    Parameters
    ----------
    trial : np.ndarray, shape as [n_sample*n_trial, len_trial, n_channel]
        trials of neural network predicitons.
    boundary : float
        decision made when any accumulated prediction surpass this boundary.

    Returns
    -------
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by bayesian inference.
    evidence : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all accumulated predictions through the process of cumsum inference.
    """
    evidence = np.cumsum(trial, axis=1, )
    response_time, choice = threshold_decision_process(evidence, boundary)
    return response_time, choice, evidence


def bayesian_inference(trial: np.ndarray, 
                       boundary: float, 
                       likelihood: np.ndarray,
                       n_channel: int):
    """
    Perform bayesian inference with neural network predictions as observations.

    Parameters
    ----------
    trial : np.ndarray, shape as [n_sample*n_trial, len_trial, n_channel]
        trials of neural network predicitons.
    boundary : float
        decision made when any posterior belief surpass this boundary.
    likelihood : np.ndarray, shape as [n_combination_possible, n_choice]
        likelihood of discretized neural network outputs in each class.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by bayesian inference.
    belief_post : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all posterior beliefs through the process of bayesian inference.
    """

    n_trial, len_trial, n_choice = trial.shape
    idx_combination = [output2index(trial[:, i], n_channel) for i in range(len_trial)]
    idx_combination = np.stack(idx_combination, axis=1)

    belief_post = np.full([n_trial, len_trial, n_choice], 1 / n_choice)
    for t in range(1, len_trial):
        # likelihood.shape = [n_choice, n_combination]
        # p(x_t|A_i) for all i(labels), shape as [n_trial, n_choice]
        prob_xt = likelihood[idx_combination[:, t]]
        # p(x_t|A_i)*p(A_i|X_1:t-1)
        prob_product = prob_xt * belief_post[:, t-1]
        belief_post[:, t] = prob_product / np.sum(prob_product, -1, keepdims=True)
    
    response_time, choice = threshold_decision_process(belief_post, boundary)
    
    return response_time, choice, belief_post


if __name__ == '__main__':
    model = load_model('imdb', 'lstm', device='cpu')
    x = torch.rand([1, 400, 300])
    y = model(x)
    pass