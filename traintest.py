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

import torch
import numpy as np
from scipy.special import softmax
from joblib import Parallel, delayed

from foolbox import PyTorchModel
from foolbox.attacks import FGSM
from foolbox.attacks import PGD
from foolbox.attacks import L2CarliniWagnerAttack as CWL2
import eagerpy as ep

from task import get_loader, get_trial, get_output_of_category, load_likelihood
from model import MNISTCNN, load_model, get_likelihood, bayesian_inference
from utils import get_pt_model, get_npz, update_trainlog


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# whether multiple GPUs exist
MULTIGPU = torch.cuda.device_count() > 1
PROPORTION = 0.1
DROPOUT = [0, 0.2, 0.4, 0.6, 0.8]
METHOD = ['clean', 'pgd']
METHOD_TITLE = ['Clean', 'PGD']
EPSILON_MNIST = [0, 0.3]
EPSILON_MNIST_FIGURE = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
EPSILON_CIFAR10 = [0, 0.03]


def train_epoch(model: torch.nn.Module, 
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
    n_train = loader.dataset.__len__()
    n_train_correct = 0
    loss_train = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        _, predicted = torch.max(output, 1)
        n_train_correct += (predicted == y).sum().item()

    loss_train = np.mean(loss_train)
    acc_train = n_train_correct / n_train

    return loss_train, acc_train


def predict_epoch(model: torch.nn.Module, 
                  loader: torch.utils.data.DataLoader, 
                  loss_func: torch.nn.modules.loss._Loss = None, 
                  return_output: bool = False,
                  device: str = DEVICE):
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
    n_sample = loader.dataset.__len__()
    n_correct = 0
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
        _, predicted = torch.max(output, 1)
        n_correct += (predicted == y).sum().item()
    loss_mean = np.mean(loss_all)
    acc = n_correct / n_sample
    
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
                device: str = DEVICE,
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
    if dataset == 'mnist' and architecture == 'cnn':
        model = MNISTCNN(dropout).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    loader_train = get_loader(dataset, subset='train', batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    loader_val = get_loader(dataset, subset='val', batch_size=batch_size, 
                            num_workers=num_workers)
    pt = get_pt_model(dataset, architecture, index, dropout)
    
    loss_val_min = 0
    idx_epoch_saved = 0
    for i in range(epochs):
        if i > idx_epoch_saved + patience:
            print('Early Stopping')
            break
        loss_train, acc_train = train_epoch(model, loader_train, loss_func, 
                                            optimizer, device)
        loss_val, acc_val = predict_epoch(model, loader_val, loss_func, device=device)
            
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
                 device: str = DEVICE,
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
    model = load_model(dataset, architecture, index, dropout_train, dropout_test,
                       device, MULTIGPU)
    x_attacked = []

    # attack CNN
    if architecture == 'cnn':
        loader = get_loader(dataset, subset=subset, proportion=proportion, 
                            batch_size=batch_size, num_workers=num_workers)
        
    # no attacks, clean evaluation
    if method == 'clean':
        acc = predict_epoch(model, loader, device=device)[-1]
    
    # attack model
    else:
        fmodel = PyTorchModel(model, bounds=(0,1))
        if method == 'fgsm':
            attack = FGSM()
        elif method == 'pgd':
            attack = PGD()
        elif method == 'cwl2':
            attack = CWL2()

        if_attacked = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x, y = ep.astensors(x, y)
            _, x_adv, success = attack(fmodel, x, y, epsilons=epsilon)
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
                device: str = DEVICE,
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
    for i in range(repeat):
        output.append(predict_epoch(model, loader, return_output=True, device=device))
    output = softmax(np.stack(output, axis=0), axis=-1)

    npz = get_npz('output', dataset, architecture, index, dropout_train, 
                  dropout_test, method, epsilon, subset, proportion, repeat)
    np.savez_compressed(npz, output=output)
    
    return output
    

def save_likelihood(dataset: str = 'mnist', 
                    architecture: str = 'cnn',
                    index: int = 0, 
                    dropout_train: float = 0, 
                    dropout_test: float = 0, 
                    method: str = 'clean', 
                    epsilon: float = 0, 
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
    method : str, optional
        adversarial attack method. The default is 'clean'.
    epsilon : float, optional
        perturbation threshold of attacks. The default is 0.
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
    args = [dataset, architecture, index, dropout_train, dropout_test, 
            method, epsilon, subset, proportion, repeat]
    output_of_category = get_output_of_category(*args)
    likelihood = get_likelihood(output_of_category, n_channel)
    npz = get_npz(*(['likelihood'] + args + [n_channel]))
    np.savez_compressed(npz, likelihood=likelihood)

    return likelihood


def save_bayes(dataset: str = 'mnist', 
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
               n_channel: int = 3,
               repeat_train: int = 10): 
    """
    Save bayesian inference results.

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
    repeat_train : int, optional
        number of neural network prediction of each training set sample. The default is 10.

    Returns
    -------
    acc : float
        mean accuracy.
    result : tuple
        (belief_post, response_time, choice).

    """
    # TODO: low acc, posterior belief rises too fast (1-2 step)
    print('Saving Bayes', dataset, architecture, index, dropout_train, dropout_test,
          method, epsilon, subset, repeat, len_trial, n_trial, boundary)
    args = [dataset, architecture, index, dropout_train, dropout_test, method, epsilon]

    args_trial = args + [subset, proportion, repeat, len_trial, n_trial]
    trial, y = get_trial(*args_trial)

    args_likelihood = args + ['train', proportion, repeat_train, n_channel]
    likelihood = load_likelihood(*args_likelihood)

    result = bayesian_inference(trial, boundary, likelihood, n_channel)
    belief_post, response_time, choice = result
    acc = np.mean(y == choice)
    print('Acc: %f'%acc)

    args_npz = args + [subset, proportion, repeat, len_trial, n_trial, boundary, n_channel]
    npz = get_npz('bayes',  *args_npz)
    np.savez_compressed(npz, acc=[acc], belief_post=belief_post,
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
        device = DEVICE
        for i in dropout:
            train_model(dataset, architecture, index, i, batch_size, epochs,
                        save, patience, device, num_workers)


def pipeline_multidropout(dataset: str = 'mnist', 
                          architecture: str = 'cnn',
                          index: int = 0, 
                          dropout: list = DROPOUT, 
                          method: str = 'clean', 
                          epsilon: float = 0, 
                          proportion_train: float = PROPORTION, 
                          repeat_train: int = 10,
                          proportion_test: float = PROPORTION,
                          repeat_test: int = 100,
                          len_trial: int = 25, 
                          n_trial: int = 10, 
                          boundary: float = 0.99, 
                          n_channel: int = 3,
                          batch_size: int = 512,
                          device: str = DEVICE,
                          num_workers: int = 4):
    """
    Convenient function to attack neural networks of multiple dropout rates.

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
    method : str, optional
        adversarial attack method. The default is 'clean'.
    epsilon : float, optional
        perturbation threshold of attacks. The default is 0.

    proportion_train : float, optional
        proportion of training set. The default is 1.
    repeat_train : int, optional
        output repeat of training set. The default is 10.
    proportion_test : float, optional
        proportion of test set. The default is 1.
    repeat_test : int, optional
        output repeat of test set. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.
    boundary : float, optional
        posterior belief decision boundary. The default is 0.99.
    n_channel : int, optional
        number of channel for likelihood estimation. The default is 3.
    batch_size : int, optional
        batch size. The default is 512.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.

    Returns
    -------
    None.

    """
    # spr stands for subset, proportion, repeat
    spr = [['train', 'test'], 
           [proportion_train, proportion_test], 
           [repeat_train, repeat_test]]

    args_likelihood = []
    args_bayes = []
    for i in dropout:
        for j in dropout:
            for s, p, r in zip(*spr):
                attack_model(dataset, architecture, index, i, j, method, epsilon,
                              s, p, batch_size, device, num_workers)
                save_output(dataset, architecture, index, i, j, method, epsilon,
                            s, p, r, batch_size, device, num_workers)
            
            args_likelihood.append([dataset, architecture, index, i, j, method, 
                                    epsilon, 'train', proportion_train, repeat_train, 
                                    n_channel])
            args_bayes.append([dataset, architecture, index, i, j, method, epsilon,
                                'test', proportion_test, repeat_test, len_trial, 
                                n_trial, boundary, n_channel, repeat_train])
    print('likelihood')
    Parallel(-1)(delayed(save_likelihood)(*i) for i in args_likelihood)
    print('bayes')
    Parallel(-1)(delayed(save_bayes)(*i) for i in args_bayes)


def pipeline_multiepsilon(dataset: str = 'mnist', 
                          architecture: str = 'cnn',
                          index: int = 0, 
                          dropout_train: float = 0, 
                          dropout_test: float = 0, 
                          method: str = 'pgd', 
                          epsilon: list = EPSILON_MNIST_FIGURE, 
                          subset: str = 'test', 
                          proportion_train: float = PROPORTION, 
                          repeat_train: int = 10,
                          proportion_test: float = PROPORTION,
                          repeat_test: int = 100,
                          len_trial: int = 25, 
                          n_trial: int = 10, 
                          boundary: float = 0.99, 
                          n_channel: int = 3,
                          batch_size: int = 512,
                          device: str = DEVICE,
                          num_workers: int = 4):
    """
    Convenient function to attack neural networks with multiple epsilons.

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
    batch_size : int, optional
        batch size. The default is 512.
    device : str, optional
        device on which the model is loaded. The default is DEVICE.
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.

    Returns
    -------
    None.

    """
    # spr stands for subset, proportion, repeat
    spr = [['train', 'test'], 
           [proportion_train, proportion_test], 
           [repeat_train, repeat_test]]

    args_likelihood = []
    args_bayes = []
    for i in epsilon:
        for s, p, r in zip(*spr):
            attack_model(dataset, architecture, index, dropout_train, dropout_test, 
                          method, i, s, p, batch_size, device, num_workers)
            save_output(dataset, architecture, index, dropout_train, dropout_test, 
                        method, i, s, p, r, batch_size, device, num_workers)
        args_likelihood.append([dataset, architecture, index, dropout_train, 
                                dropout_test, method, i, 'train', proportion_train, 
                                repeat_train, n_channel])
        args_bayes.append([dataset, architecture, index, dropout_train, dropout_test, 
                           method, i, 'test', proportion_test, repeat_test, len_trial, 
                           n_trial, boundary, n_channel, repeat_train])
    Parallel(-1)(delayed(save_likelihood)(*i) for i in args_likelihood)
    Parallel(-1)(delayed(save_bayes)(*i) for i in args_bayes)
    pass

    

if __name__ == '__main__':
    time0 = time()
    # for i in DROPOUT:
        # train_model(dropout=i)
    # pipeline_multidropout()
    # pipeline_multidropout(method='pgd', epsilon=0.3)
    # pipeline_multiepsilon()
    # pipeline_multiepsilon(dropout_train=0.2, dropout_test=0.6)
    # MULTIGPU = True
    train_model_multidropout(dropout=[0])
    print('Cost %.2f mins'%((time() - time0) / 60))