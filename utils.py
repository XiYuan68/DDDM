#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:45:32 2021

@author: chenxiyuan

utility functions, mainly getters of file paths.
"""

from os import listdir, mkdir
from datetime import datetime
from shutil import copyfile


def get_dir(dataset: str = 'mnist', subdir: str = 'model'):
    """
    Return path of subdir in ./data/.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    subdir : str, optional
        name of sub-directory. The default is 'model'.

    Returns
    -------
    path : str
        path of subdir in ./data/.

    """
    path = '/'.join(['data', dataset, subdir, ''])
    return path


def mkdir_databackup():
    """
    Create empty directories of ./data/ and ./backup/ if not existing.

    Returns
    -------
    None.

    """
    path = listdir()
    if 'data' not in path:
        mkdir('data')
    if 'backup' not in path:
        mkdir('backup')


def mkdir_dataset(dataset: str = 'mnist'):
    """
    Create empty directories in ./data/ for data storage.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.

    Returns
    -------
    None.

    """
    path = listdir('data')
    if dataset not in path:
        print('Creating Directories for', dataset)
        dir_dataset = 'data/'+dataset
        mkdir(dir_dataset)
        subdir = ['model', 'figure', 'attack', 'output', 'likelihood', 'bayes', 
                  'others', 'cumsum']
        for i in subdir:
            mkdir(get_dir(dataset, i))


def get_pt_model(dataset: str = 'mnist', 
                 architecture: str = 'cnn', 
                 index: int = 0, 
                 dropout: float = 0):
    """
    Return path of pytorch model `.pt` file.

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

    Returns
    -------
    pt : str
        path of pytorch model `.pt` file.

    """
    d = get_dir(dataset, 'model')
    pt = d + '_'.join([architecture, str(index), str(dropout)]) + '.pt'
    return pt


def get_txt_trainlog(dataset: str = 'mnist', 
                     architecture: str = 'cnn', 
                     index: int = 0, 
                     dropout: float = 0):
    """
    Return path of model training log `.txt` file.

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

    Returns
    -------
    txt : str
        path of model training log `.txt` file.

    """
    d = get_dir(dataset, 'model')
    txt = d + '_'.join([architecture, str(index), str(dropout)]) + '.txt'
    return txt


def update_trainlog(dataset: str = 'mnist', 
                    architecture: str = 'cnn', 
                    index: int = 0, 
                    dropout: float = 0,
                    line: str = '', 
                    mode: str = 'a'):
    """
    Write meassage in training log.

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
    line : str, optional
        training message to be looged. The default is ''.
    mode : str, optional
        mode of opening file. The default is 'a'.

    Returns
    -------
    None.

    """
    txt = get_txt_trainlog(dataset, architecture, index, dropout)
    with open(txt, mode) as f:
        f.write(line + '\n')


def args2npz(dataset: str, subdir: str, args: list):
    """
    Return path of `.npz` file

    Parameters
    ----------
    dataset : str
        name of dataset.
    subdir : str
        name of sub-directory.
    args : list
        other arguments of desired `.npz` file.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    d = get_dir(dataset, subdir)
    npz = '_'.join([str(i) for i in args])
    npz = d + npz + '.npz'
    return npz 


def get_npz_attack(dataset: str,
                   architecture: str, 
                   index: int, 
                   dropout_train: float, 
                   dropout_test: float, 
                   method: str, 
                   epsilon: float, 
                   subset: str, 
                   proportion: float):
    """
    Return path of `.npz` file that contains adversarial samples.

    Parameters
    ----------
    dataset : str
        name of dataset.
    architecture : str
        architecture of the neural network.
    index : int
        index of data to return.
    dropout_train : float
        dropout rate at training phase.
    dropout_test : float
        dropout rate at test phase.
    method : str
        adversarial attack method.
    epsilon : float
        perturbation threshold of attacks.
    subset : str
        subset of dataset.
    proportion : float
        proportion of subset data.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    args = ['attack', architecture, index, dropout_train, dropout_test, method, epsilon, 
            subset, proportion]
    npz = args2npz(dataset, 'attack', args)
    return npz


def get_npz_output(dataset: str, 
                   architecture: str, 
                   index: int, 
                   dropout_train: float, 
                   dropout_test: float, 
                   method: str, 
                   epsilon: float, 
                   subset: str, 
                   proportion: float, 
                   repeat: int):
    """
    Return path of `.npz` file that contains outputs of neural networks.

    Parameters
    ----------
    dataset : str
        name of dataset.
    architecture : str
        architecture of the neural network.
    index : int
        index of data to return.
    dropout_train : float
        dropout rate at training phase.
    dropout_test : float
        dropout rate at test phase.
    method : str
        adversarial attack method.
    epsilon : float
        perturbation threshold of attacks.
    subset : str
        subset of dataset.
    proportion : float
        proportion of subset data.
    repeat : int
        number of neural network prediction of each sample.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    args = ['output', architecture, index, dropout_train, dropout_test, method, epsilon, 
            subset, proportion, repeat]
    npz = args2npz(dataset, 'output', args)
    return npz


def get_npz_likelihood(dataset: str, 
                       architecture: str, 
                       index: int, 
                       dropout_train: float, 
                       dropout_test: float, 
                       method: str, 
                       epsilon: float, 
                       subset: str, 
                       proportion: float, 
                       repeat: int, 
                       n_channel: int):
    """
    Return path of `.npz` file that contains estimated likelihood.

    Parameters
    ----------
    dataset : str
        name of dataset.
    architecture : str
        architecture of the neural network.
    index : int
        index of data to return.
    dropout_train : float
        dropout rate at training phase.
    dropout_test : float
        dropout rate at test phase.
    method : str
        adversarial attack method.
    epsilon : float
        perturbation threshold of attacks.
    subset : str
        subset of dataset.
    proportion : float
        proportion of subset data.
    repeat : int
        number of neural network prediction of each sample.
    n_channel : int
        number of channel in outputs to used for likelihood estimation.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    args = ['likelihood', architecture, index, dropout_train, dropout_test, method, 
            epsilon, subset, proportion, repeat, n_channel]
    npz = args2npz(dataset, 'likelihood', args)
    return npz


def get_npz_bayes(dataset: str, 
                  architecture: str, 
                  index: int, 
                  dropout_train: float, 
                  dropout_test: float, 
                  method: str, 
                  epsilon: float, 
                  subset: str, 
                  proportion: float, 
                  repeat: int, 
                  len_trial: int, 
                  n_trial: int, 
                  boundary: float, 
                  n_channel: int,
                  likelihood_method: list,
                  likelihood_epsilon: list):
    """
    Return path of `.npz` file that contains bayesian inference results.

    Parameters
    ----------
    dataset : str
        name of dataset.
    architecture : str
        architecture of the neural network.
    index : int
        index of data to return.
    dropout_train : float
        dropout rate at training phase.
    dropout_test : float
        dropout rate at test phase.
    method : str
        adversarial attack method.
    epsilon : float
        perturbation threshold of attacks.
    subset : str
        subset of dataset.
    proportion : float
        proportion of subset data.
    repeat : int
        number of neural network prediction of each sample.
    len_trial : int
        length of each trial.
    n_trial : int
        number of trials for each sample.
    boundary : float
        posterior belief decision boundary.
    n_channel : int
        number of channel for likelihood estimation.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    args = ['bayes', architecture, index, dropout_train, dropout_test, method, 
            epsilon, subset, proportion, repeat, len_trial, n_trial, boundary, 
            n_channel, likelihood_method, likelihood_epsilon]
    npz = args2npz(dataset, 'bayes', args)
    return npz


def get_npz_cumsum(dataset: str, 
                   architecture: str, 
                   index: int, 
                   dropout_train: float, 
                   dropout_test: float, 
                   method: str, 
                   epsilon: float, 
                   subset: str, 
                   proportion: float, 
                   repeat: int, 
                   len_trial: int, 
                   n_trial: int, 
                   boundary: float):
    """
    Return path of `.npz` file that contains bayesian inference results.

    Parameters
    ----------
    dataset : str
        name of dataset.
    architecture : str
        architecture of the neural network.
    index : int
        index of data to return.
    dropout_train : float
        dropout rate at training phase.
    dropout_test : float
        dropout rate at test phase.
    method : str
        adversarial attack method.
    epsilon : float
        perturbation threshold of attacks.
    subset : str
        subset of dataset.
    proportion : float
        proportion of subset data.
    repeat : int
        number of neural network prediction of each sample.
    len_trial : int
        length of each trial.
    n_trial : int
        number of trials for each sample.
    boundary : float
        posterior belief decision boundary.


    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    args = ['cumsum', architecture, index, dropout_train, dropout_test, method, 
            epsilon, subset, proportion, repeat, len_trial, n_trial, boundary]
    npz = args2npz(dataset, 'cumsum', args)
    return npz


def get_npz(mode: str, *args: list):
    """
    Convenient function to return path of `.npz` file.

    Parameters
    ----------
    mode : str
        type of `.npz` path to return, one of [ attack | output | likelihood | bayes ].
    *args : list
        other arguments of desired `.npz` file, depend on `get_npz_mode()`.

    Returns
    -------
    npz : str
        path of `.npz` file.

    """
    string_func = '_'.join(['get_npz', mode])
    npz = eval(string_func)(*args)
    return npz


def backup_script():
    """
    Copy all .py files to a time-tagged directory in backup/ .

    Returns
    -------
    None.

    """
    script = [i for i in listdir() if '.py' in i]
    backup_dir = 'backup/' + str(datetime.now())
    mkdir(backup_dir)
    [copyfile(i, '/'.join([backup_dir, i])) for i in script]
    print('all scripts back-upped at', backup_dir)


# create empty directories ./data/ and ./backup/ if not existing
mkdir_databackup()


if __name__ == '__main__':
    backup_script()
    pass
