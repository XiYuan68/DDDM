#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:29:39 2021

@author: chenxiyuan

train, test and attack neural networks, estimate likelihood and perform bayesian
inference.
"""
from time import time
import subprocess
from itertools import permutations
from typing import Union
from copy import deepcopy

import torch
import numpy as np
from scipy.special import softmax
from joblib import Parallel, delayed

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from foolbox import PyTorchModel
from foolbox.attacks import (FGSM, PGD, L2DeepFoolAttack, SpatialAttack,
                             SaltAndPepperNoiseAttack, LinfRepeatedAdditiveUniformNoiseAttack)
import eagerpy as ep

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SquareAttack

from task import (get_loader, get_trial, get_output_of_category, load_likelihood,
                  DATASET_TEXT, DATASET_AUDIO, iter2array)
from model import (MNISTCNN, IMDBLSTM, CIFAR10RESNET, SPEECHCOMMANDSDEEPSPEECH,
                   load_model, get_likelihood, 
                   bayesian_inference, cumsum_inference)
from utils import get_pt_model, get_npz, update_trainlog


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# whether multiple GPUs exist
MULTIGPU = torch.cuda.device_count() > 1
PROPORTION = 0.1
DROPOUT = [0, 0.2, 0.4, 0.6, 0.8]

METHOD_MNIST = ['clean', 'fgsm', 'pgd', 'cwl2', 'deepfool', 
                'spatial', 'saltpepper', 'uniform', 'square']
EPSILON_MNIST = [0, 0.3, 0.3, 0, 1.5,    0, 1.5, 0.3, 0.3]
EPSILON_MNIST_FIGURE = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

METHOD_CIFAR10 = ['clean', 'pgd']
EPSILON_CIFAR10 = [0, 0.031]
EPSILON_CIFAR10_FIGURE = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

METHOD_IMDB = ['clean', 'textbugger']
EPSILON_IMDB = [0, 0]

METHOD_SPEECHCOMMANDS = ['clean', 'imperceptible']
EPSILON_SPEECHCOMMANDS = [0, 0.05]

METHOD_TITLE = {'clean': 'Clean', 'pgd': 'PGD', 'cwl2': 'C&W L2', 
                'deepfool': 'DeepFool L2', 'saltpepper': 'Salt&Pepper',
                'uniform': 'Uniform Linf', 'square': 'Square Linf',
                'spatial': 'Spatial', 'textbugger': 'TextBugger',
                'imperceptible': 'Imperceptible'}


class AccuracyGetter:
    def __init__(self, dataset: str, architecture: str):
        if dataset == 'mnist' and architecture == 'cnn':
            self.feature_dim = 1
        elif dataset == 'imdb' and architecture == 'lstm':
            self.feature_dim = -1   
        elif dataset == 'cifar10' and architecture == 'resnet':
            self.feature_dim = 1
        elif dataset == 'speechcommands' and architecture == 'deepspeech':
            self.feature_dim = -1
        else:
            raise Exception('Wrong dataset and architecture combination: %s + %s'%(dataset, architecture))
        self.n_sample = 0
        self.n_correct = 0

    def compute_batch(self, output: torch.Tensor, y: torch.Tensor):
        # RNN return sequence instead of single label
        if output.ndim > 2:
            output = torch.mean(output, dim=1)
        self.n_sample += y.shape[0]
        _, predicted = torch.max(output, self.feature_dim)
        self.n_correct += (predicted == y).sum().item()

    def get_accuracy(self):
        acc = self.n_correct / self.n_sample
        
        return acc


def train_epoch(dataset: str,
                architecture: str, 
                model: torch.nn.Module, 
                loader: torch.utils.data.DataLoader, 
                loss_func: torch.nn.modules.loss._Loss, 
                optimizer: torch.optim.Optimizer,
                device: str):
    """
    Train given model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        neural network classifier.
    loader : torch.utils.data.DataLoader
        pytorch data loader.
    loss_func : torch.nn.modules.loss._Loss
        pytorch loss function.
    optimizer : torch.optim.Optimizer
        pytorch optimizer.
    device : str, optional
        device on which the model is loaded.

    Returns
    -------
    loss_train : float
        mean loss in this epoch.
    acc_train : float
        mean accuracy in this epoch..

    """
    model.train()
    acc_getter = AccuracyGetter(dataset, architecture)
    loss_train = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        acc_getter.compute_batch(output, y)

    loss_train = np.mean(loss_train)
    acc_train = acc_getter.get_accuracy()

    return loss_train, acc_train


def predict_epoch(dataset: str,
                  architecture: str, 
                  model: torch.nn.Module, 
                  loader: torch.utils.data.DataLoader, 
                  loss_func: torch.nn.modules.loss._Loss = None, 
                  return_output: bool = False,
                  device: torch.device = DEVICE):
    """
    Let given neural network make predicitons.

    Parameters
    ----------
    model : torch.nn.Module
        neural network classifier.
    loader : torch.utils.data.DataLoader
        pytorch data loader.
    loss_func : torch.nn.modules.loss._Loss, optional
        pytorch loss function. The default is None.
    return_output : bool, optional
        whether return model prediction. The default is False.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.

        when False, mean loss and accuracy of this epoch will be returned.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    model.eval()
    acc_getter = AccuracyGetter(dataset, architecture)
    loss_all = []
    output_all = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if loss_func is not None:
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)
        output = model(x)
        if return_output:
            output_all.append(output.cpu().detach().numpy())
        if loss_func is not None:
            loss = loss_func(output, y)
            loss_all.append(loss.item())
        acc_getter.compute_batch(output, y)

    loss_mean = np.mean(loss_all)
    acc = acc_getter.get_accuracy()
    
    if return_output:
        output_all = np.concatenate(output_all, -2)
        return output_all
    else:
        return loss_mean, acc


def train_model(dataset='mnist', 
                architecture: str = 'cnn', 
                index: int = 0, 
                dropout: float = 0, 
                batch_size: int = 64, 
                epochs: int = 50, 
                save: bool = True, 
                patience: int = 20,
                device: torch.device = DEVICE,
                num_workers: int = 4):
    """
    Train a neural network.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout : float, optional
        dropout rate at training phase. The default is 0.
    batch_size : int, optional
        batch size. The default is 64.
    epochs : int, optional
        number of maximum training epochs. The default is 50.
    save : bool, optional
        whether save model after training. The default is True.
    patience : int, optional
        training will be stopped if validation loss stops decreasing in *patience* epochs. 
        The default is 20.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.

    Returns
    -------
    model : torch.nn.Module
        well-trained neural network.

    """

    print('Training', dataset, architecture, index, dropout)
    
    update_trainlog(dataset, architecture, index, dropout, mode='w')
    model = load_model(dataset, architecture, index, dropout, None,
                       device, return_untrained=True)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    loader_train = get_loader(dataset, subset='train', batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, 
                              phase='training')
    loader_val = get_loader(dataset, subset='val', batch_size=batch_size, 
                            num_workers=num_workers)
    pt = get_pt_model(dataset, architecture, index, dropout)
    
    loss_val_min = 0
    idx_epoch_saved = 0
    for i in range(epochs):
        if i > idx_epoch_saved + patience:
            print('Early Stopping')
            break
        loss_train, acc_train = train_epoch(dataset, architecture, 
                                            model, loader_train, loss_func, 
                                            optimizer, device)
        loss_val, acc_val = predict_epoch(dataset, architecture, 
                                          model, loader_val, loss_func, 
                                          device=device)
            
        line = 'Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}, Val Loss: {:.4f}, Val Acc: {:.2f}'.format(i+1, epochs, loss_train, acc_train, loss_val, acc_val)
        print(line)
        update_trainlog(dataset, architecture, index, dropout, line)
        # save current model if it shows lowest val loss in history
        if save and (i == 0 or loss_val < loss_val_min):
            line = 'Saving model'
            print(line)
            update_trainlog(dataset, architecture, index, dropout, line)
            loss_val_min = loss_val
            torch.save(model.state_dict(), pt)
            idx_epoch_saved = i
    
    return model


def attack_model(dataset: str = 'mnist', 
                 architecture: str = 'cnn',
                 index: int = 0, 
                 dropout_train: float = 0, 
                 dropout_test: float = 0, 
                 method: str = 'clean', 
                 epsilon: float = 0, 
                 subset: str = 'test', 
                 proportion: float = PROPORTION, 
                 batch_size: int = 512, 
                 save: bool = True,
                 device: torch.device = DEVICE,
                 num_workers: int = 4):
    """
    Attack neural network if method!='clean'.

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
    batch_size : int, optional
        batch size. The default is 512.
    save : bool, optional
        whether save generated adversarial samples. The default is True.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.

    Returns
    -------
    x_attacked : np.ndarray, shape as [n_sample, *input_shape]
        adversarial samples.

    """
    print('Attacking', dataset, architecture, index, dropout_train, dropout_test, 
          method, epsilon, subset, proportion)
    if dataset in DATASET_TEXT+DATASET_AUDIO and method != 'clean':
        print('Skipping')
        return 0
        
    
    model = load_model(dataset, architecture, index, dropout_train, dropout_test,
                       device, MULTIGPU)
    x_attacked = []

    phase = None if method == 'clean' else 'attacking'
    loader = get_loader(dataset, subset=subset, proportion=proportion, 
                        batch_size=batch_size, num_workers=num_workers, 
                        phase=phase)
        
        
    # no attacks, clean evaluation
    if method == 'clean':
        acc = predict_epoch(dataset, architecture, model, loader, device=device)[-1]
    
    # attack model
    else:
        if method == 'square':
            x = torch.cat([x for x, y in loader], dim=0)
            x = iter2array(x)
            y = iter2array(loader.dataset.targets)
            
            n_class = 10
            # mnist or cifar10
            input_shape = (1, 28, 28) if dataset == 'mnist' else (3, 32, 32)
            classifier = PyTorchClassifier(model, torch.nn.CrossEntropyLoss(),
                                           input_shape, n_class, 
                                           clip_values=(0, 1))
            attack = SquareAttack(classifier, eps=epsilon, batch_size=batch_size)
            x_attacked = attack.generate(x=x)
            pred = classifier.predict(x_attacked, batch_size=batch_size)
            acc = np.mean(np.argmax(pred, axis=-1) == y)
        
        else:
            if dataset == 'mnist':
                preprocessing = None
            elif dataset == 'cifar10':
                preprocessing = dict(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],
                                     axis=-3)
            else:
                raise Exception('Wrong dataset: %s'%dataset)
    
            if method == 'pgd':
                attack = PGD()
            # here we use the cleverhans C&W L2 implemantation, which is faster and 
            # has higher success rate
            elif method == 'cwl2':
                pass
            elif method == 'spatial':
                attack = SpatialAttack()
            elif method == 'fgsm':
                attack = FGSM()
            elif method == 'deepfool':
                attack = L2DeepFoolAttack()
            elif method == 'saltpepper':
                attack = SaltAndPepperNoiseAttack()
            elif method == 'uniform':
                attack = LinfRepeatedAdditiveUniformNoiseAttack()
            else:
                raise Exception('Invalid attack method: %s'%method)
    
    
            fmodel = PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)
    
    
            if_attacked = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if method != 'cwl2':
                    x, y = ep.astensors(x, y)
                    _, x_adv, success = attack(fmodel, x, y, epsilons=epsilon)
                else:
                    # TODO: maybe there should be some preprocessing here 
                    # if attacking resnet
                    x_adv = carlini_wagner_l2(model, x, 10)
                    pred_adv = torch.argmax(model(x_adv), axis=-1)
                    x_adv = x_adv.cpu()
                    success = (pred_adv != y).cpu()
                x_attacked.append(x_adv.numpy())
                if_attacked += list(success.numpy())
            
            x_attacked = np.concatenate(x_attacked)
            acc = 1 - np.mean(if_attacked)


    print('Acc: %f'%acc)
    if save:
        npz = get_npz('attack', dataset, architecture, index, dropout_train, 
                      dropout_test, method, epsilon, subset, proportion)
        np.savez_compressed(npz, x=x_attacked, acc=[acc])
    
    return x_attacked


def save_output(dataset: str = 'mnist', 
                architecture: str = 'cnn',
                index: int = 0, 
                dropout_train: float = 0, 
                dropout_test: float = 0, 
                method: str = 'clean', 
                epsilon: float = 0, 
                subset: str = 'test', 
                proportion: float = PROPORTION, 
                repeat: int = 100,
                batch_size: int = 1024,
                device: torch.device = DEVICE,
                num_workers: int = 4):
    """
    Save neural network output.

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
    batch_size : int, optional
        batch size. The default is 1024.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.

    Returns
    -------
    output : np.ndarray, shape as [n_sample, n_choice]
        neural network classifier outputs.

    """
    print('Saving Output', dataset, architecture, index, dropout_train, dropout_test, 
          method, epsilon, subset, proportion, repeat)
    
    loader = get_loader(dataset, architecture, index, dropout_train, dropout_test,
                        method, epsilon, subset, proportion, batch_size, num_workers)

    model = load_model(dataset, architecture, index, dropout_train, dropout_test,
                       device, MULTIGPU)

    output = []
    # print(predict_epoch(dataset, architecture, model, loader, device=device))
    for i in range(repeat):
        output.append(predict_epoch(dataset, architecture, model, loader, 
                                    return_output=True, device=device))
    output_raw = np.stack(output, axis=0)
    output = softmax(output_raw, axis=-1)
        
    npz = get_npz('output', dataset, architecture, index, dropout_train, 
                  dropout_test, method, epsilon, subset, proportion, repeat)
    np.savez_compressed(npz, output=output, output_raw=output_raw)
    
    return output
    

def save_likelihood(dataset: str = 'mnist', 
                    architecture: str = 'cnn',
                    index: int = 0, 
                    dropout_train: float = 0, 
                    dropout_test: float = 0, 
                    method: list = ['clean', 'pgd'], 
                    epsilon: list = [0, 0.3], 
                    subset: str = 'train', 
                    proportion: float = PROPORTION, 
                    repeat: int = 10,
                    n_channel: int = 3):
    """
    Save estimated likelihood of discretized neural network outputs in each class.

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
    method : list, optional
        attack methods. The default is ['clean', 'pgd'].
    epsilon : list, optional
        perturbation threshold of attacks. The default is [0, 0.3].
    subset : str, optional
        subset of dataset. The default is 'train'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 10.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.

    Returns
    -------
    likelihood : TYPE
        DESCRIPTION.

    """
    print('Saving Likelihood', dataset, architecture, index, dropout_train, 
          dropout_test, method, epsilon, subset, proportion, repeat, n_channel)
    output_of_category_permethod = []
    for m, e in zip(method, epsilon):
        args = [dataset, architecture, index, dropout_train, dropout_test, 
                m, e, subset, proportion, repeat]
        output_of_category_permethod.append(get_output_of_category(*args))
    output_of_category = []
    n_choice = len(output_of_category_permethod[0])
    for i in range(n_choice):
        output_of_category.append(np.concatenate([j[i] for j in output_of_category_permethod]))
    likelihood = get_likelihood(output_of_category, n_channel)
    args = [dataset, architecture, index, dropout_train, dropout_test, 
            method, epsilon, subset, proportion, repeat]
    npz = get_npz(*(['likelihood'] + args + [n_channel]))
    np.savez_compressed(npz, likelihood=likelihood)

    return likelihood


def save_inference(dataset: str = 'mnist', 
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
                   inference: str = 'bayes',
                   boundary: float = 0.99,
                   n_channel: int = 3,
                   repeat_train: int = 10,
                   likelihood_method: list = ['clean'],
                   likelihood_epsilon: list = [0]): 
    """
    Save cumsum/bayesian inference results.

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
    inference : str, optional
        type of inference to perform, choose from ['cumsum' | 'bayes']. 
        The default is 'bayes'.
    boundary : float, optional
        decision evidence boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation, only for bayes inference. 
        The default is 3.
    repeat_train : int, optional
        number of neural network prediction of each training set sample to 
        estimate likelihood, only for bayes inference.
        The default is 10.
    likelihood_method : list, optional
        strings of attack methods for likelihood estimation, only for bayes inference. 
        The default is ['clean'].
    likelihood_epsilon : list, optional
        floats of attack epsilons for likelihood estimation, only for bayes inference. 
        The default is [0].
        

    Returns
    -------
    acc : float
        mean accuracy.
    result : tuple
        (belief_post, response_time, choice).

    """
    # TODO: low acc, posterior belief rises too fast (1-2 step)
    print('Saving', inference, dataset, architecture, index, dropout_train, 
          dropout_test, method, epsilon, subset, repeat, len_trial, n_trial, 
          boundary, )
    args = [dataset, architecture, index, dropout_train, dropout_test, method, epsilon]

    args_trial = args + [subset, proportion, repeat, len_trial, n_trial]
    trial, y = get_trial(*args_trial)

    args_npz = args + [subset, proportion, repeat, len_trial, n_trial, boundary]
    if inference == 'bayes':
        args_likelihood = [dataset, architecture, index, dropout_train, dropout_test]
        args_likelihood += [likelihood_method, likelihood_epsilon, 'train', 
                            proportion, repeat_train, n_channel]
        likelihood = load_likelihood(*args_likelihood)
        result = bayesian_inference(trial, boundary, likelihood, n_channel)
        args_npz += [n_channel, likelihood_method, likelihood_epsilon]
    elif inference == 'cumsum':
        result = cumsum_inference(trial, boundary)
    else:
        raise Exception("Invalid inference: %s"%inference)

    response_time, choice, evidence = result
    acc = np.mean(y == choice)
    print('Acc: %f'%acc)
    npz = get_npz(inference,  *args_npz)
    np.savez_compressed(npz, acc=[acc], evidence=evidence,
                        response_time=response_time, choice=choice)
    
    return acc, result


def train_model_multidropout(dataset='mnist', 
                             architecture: str = 'cnn', 
                             index: int = 0, 
                             dropout: list = DROPOUT, 
                             batch_size: int = 64, 
                             epochs: int = 50, 
                             save: bool = True, 
                             patience: int = 20,
                             device: list = list(range(len(DROPOUT))),
                             num_workers: int = 4):
    """
    Train multiple models with multiple dropout rates 
    (on multiple GPUs simultaneously if available).

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
    batch_size : int, optional
        batch size. The default is 64.
    epochs : int, optional
        number of maximum training epochs. The default is 50.
    save : bool, optional
        whether save model after training. The default is True.
    patience : int, optional
        training will be stopped if validation loss stops decreasing in *patience* epochs. 
        The default is 20.
    device : list of ints, optional
        indexs of GPUs on which the models are loaded.
        The default is None, same as list(range(len(dropout))).
    num_workers: int, optional
        how many subprocesses to use for data loader. The default is 4.

    Returns
    -------
    None.

    """
    # train models on multiple GPUs simultaneously
    if MULTIGPU:
        if device is None:
            device = list(range(len(dropout)))
        error = 'number of GPUs (%d) != number of dropout rates (%d)'%(len(device), len(dropout))
        assert len(device) == len(dropout), error
        device = ['cuda:'+str(i) for i in device]
        # TODO: only training progress of last model can be output
        # it's ideal to show progress of all training
        # create a bash script to run multiple train_single_model.py simultaneously
        # the script be like:
            # #!/bin/bash
            # python train_single_model.py ... dropout 0 ... --device cuda:0 &
            # python train_single_model.py ... dropout 0.2 ... --device cuda:1 &
            # python train_single_model.py ... dropout 0.4 ... --device cuda:2
        lines = ['#!/bin/bash\n']
        for dr, de in zip(dropout, device):
            l = 'python train_single_model.py '
            arg_str = ['dataset', 'architecture', 'index', 'dropout', 'batch_size', 
                       'epochs', 'save', 'patience', 'device', 'num_workers']
            arg_str = ['--'+i+' ' for i in arg_str]
            arg_arg = [dataset, architecture, index, dr, batch_size, epochs,
                       save, patience, de, num_workers]
            l += ' '.join([s+str(a) for s, a in zip(arg_str, arg_arg)]) + ' &\n'
            lines.append(l)
        # print training progress of last model in terminal
        lines[-1] = l[:-3]
        script = 'train_multiple_model.sh'
        with open(script, 'w') as f:
            f.writelines(lines)
        # run the bash script
        subprocess.run(['bash', script])
        pass

    # train models on single GPU
    else:
        device = 'cuda'
        for i in dropout:
            train_model(dataset, architecture, index, i, batch_size, epochs,
                        save, patience, device, num_workers)


def get_dropout_pair(dropout_train: list = DROPOUT, dropout_test: list = DROPOUT):
    """
    Return cartesian product of given training and testing dropout rates.

    Parameters
    ----------
    dropout_train : list, optional
        training dropout rates. The default is DROPOUT.
    dropout_test : list, optional
        testing dropout rates. The default is DROPOUT.

    Returns
    -------
    dropout_pair : list
        training and testing dropout rate pairs.

    """
    dropout_pair = []
    for i in dropout_train:
        for j in dropout_test:
            dropout_pair.append((i, j))
    return dropout_pair


def pipeline(dataset: str = 'mnist', 
             architecture: str = 'cnn',
             index: int = 0, 
             dropout_pair: Union[list, tuple] = (0, 0), 
             method: str = 'clean', 
             epsilon: Union[float, int, list] = 0, 
             proportion_train: float = PROPORTION, 
             repeat_train: int = 10,
             proportion_test: float = PROPORTION,
             repeat_test: int = 100,
             len_trial: int = 25, 
             n_trial: int = 10, 
             boundary_cumsum: float = 5, 
             boundary_bayes: float = 0.99, 
             n_channel: int = 3,
             batch_size: int = 512,
             device: torch.device = DEVICE,
             num_workers: int = 4,
             likelihood_method: list = ['clean'],
             likelihood_epsilon: list = [0],
             likelihood_estimate: bool = True):
    """
    Convenient function to attack neural networks, estimate likelihood and 
    perform cumsum and bayesian inference.
    
    Cumsum inference, likelihood estimation and bayesian inference will be 
    processed in a CPU parallel fashion if multiple dropout rate pairs and/or
    multiple epsilon values are given.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    architecture : str, optional
        architecture of the neural network. The default is 'cnn'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_pair : Union[list, tuple], optional
        training and testing dropout rate pair(s). 
        single dropout rate pair should be in the form as (d_train, d_test).
        multiple dropout rate pairs should be in the form as 
        [(d_train_0, d_test_0), (d_train_1, d_test_1), (d_train_2, d_test_2), ...],
        which could be returned by get_dropout_pair().
        The default is (0, 0).
    method : str, optional
        adversarial attack method. The default is 'clean'.
    epsilon : Union[float, int, list], optional
        perturbation threshold(s) of attacks. The default is 0.
    proportion_train : float, optional
        proportion of training set. The default is 1.
    repeat_train : int, optional
        prediction repeat of training set. The default is 10.
    proportion_test : float, optional
        proportion of test set. The default is 1.
    repeat_test : int, optional
        prediction repeat of test set. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.
    boundary_cumsum : float, optional
        neural network prediction decision boundary. The default is 5.
    boundary_bayes : float, optional
        posterior belief decision boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.
    batch_size : int, optional
        batch size. The default is 512.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.
    likelihood_method : list, optional
        strings of attack methods for likelihood estimation, only for bayes inference. 
        The default is ['clean'].
    likelihood_epsilon : list, optional
        floats of attack epsilons for likelihood estimation, only for bayes inference. 
        The default is [0].
    likelihood_estimate : bool, optional
        whether estimate likelihood. The default is True.

    Returns
    -------
    None.

    """

    if likelihood_method == ['clean'] and method != 'clean':
        spr = [['test'], 
               [proportion_test], 
               [repeat_test]]
    else:
        spr = [['train', 'test'], 
               [proportion_train, proportion_test], 
               [repeat_train, repeat_test]]        

    args_likelihood = []
    args_cumsum = []
    args_bayes = []
    
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = [epsilon]        
    if isinstance(dropout_pair, tuple):
        dropout_pair = [dropout_pair]

    for i, j in dropout_pair:
        for e in epsilon:
            for s, p, r in zip(*spr):
                attack_model(dataset, architecture, index, i, j, method, e, 
                             s, p, batch_size, True, device, num_workers)
                save_output(dataset, architecture, index, i, j, method, e, 
                            s, p, r, batch_size, device, num_workers)
            
            args_cumsum.append([dataset, architecture, index, i, j, method, 
                                e, 'test', proportion_test, repeat_test, 
                                len_trial, n_trial, 'cumsum', boundary_cumsum])
            args_likelihood.append([dataset, architecture, index, i, 
                                    j, likelihood_method, likelihood_epsilon, 
                                    'train', proportion_train, repeat_train, 
                                    n_channel])
            args_bayes.append([dataset, architecture, index, i, j, method, 
                               e, 'test', proportion_test, repeat_test, 
                               len_trial, n_trial, 'bayes', boundary_bayes, 
                               n_channel, repeat_train, likelihood_method, 
                               likelihood_epsilon])
    Parallel(-1)(delayed(save_inference)(*i) for i in args_cumsum)
    if likelihood_estimate:
        Parallel(-1)(delayed(save_likelihood)(*i) for i in args_likelihood)
    Parallel(-1)(delayed(save_inference)(*i) for i in args_bayes)

    

if __name__ == '__main__':
    time0 = time()
    dataset = 'mnist'
    architecture = 'cnn'
    index = 0
    proportion_train = 0.1
    proportion_test = 0.1
    batch_size = 512
    n_channel = 3
    args = []
    lm = ['clean']
    le = [0]
    method = 'square'
    epsilon = EPSILON_MNIST[METHOD_MNIST.index(method)]
    attack_model(dataset, architecture, method=method, epsilon=epsilon)
    # pipeline()
    # pipeline(method='pgd', epsilon=0.3)
    # pipeline(method='cwl2', epsilon=1.5)
    # pipeline(method='spatial', epsilon=0)
    # for i in DROPOUT:
    #     train_model(dataset, architecture, index, dropout=i,)
    
    # dropout_pair = get_dropout_pair([0.8], DROPOUT)
    # for m, e in zip(['cwl2'], [0]):

    #     pipeline(dataset, architecture, index, get_dropout_pair(), m, e, 
    #              proportion_train=proportion_train, 
    #              proportion_test=proportion_test,
    #              n_channel=n_channel,
    #              batch_size=batch_size,
    #              likelihood_method=lm, 
    #              likelihood_epsilon=le)
    # i = 0
    # j = 0.8
    # m = 'pgd'
    # e = deepcopy(EPSILON_MNIST_FIGURE)
    # e.remove(0.3)
    # pipeline_parallel(dataset, architecture, index, (i, j), m, e, 
    #                   proportion_train=proportion_train, 
    #                   proportion_test=proportion_test,
    #                   n_channel=n_channel,
    #                   batch_size=batch_size,
    #                   likelihood_method=lm, 
    #                   likelihood_epsilon=le)

    print('Cost %.2f mins'%((time() - time0) / 60))