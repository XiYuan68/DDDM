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
    path = '/'.join(['data', dataset, subdir, ''])
    return path


def mkdir_dataset(dataset: str = 'mnist'):
    print('Creating Directories for', dataset)
    dir_dataset = 'data/'+dataset
    mkdir(dir_dataset)
    subdir = ['model', 'figure', 'attack', 'output', 'likelihood', 'bayes', 'others']
    for i in subdir:
        mkdir(get_dir(dataset, i))


def get_pt_model(dataset: str = 'mnist', 
                 architecture: str = 'cnn', 
                 index: int = 0, 
                 dropout: float = 0):
    d = get_dir(dataset, 'model')
    pt = d + '_'.join([architecture, str(index), str(dropout)]) + '.pt'
    return pt


def args2npz(dataset: str, subdir: str, args: list):
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
                  n_channel: int):
    args = ['bayes', architecture, index, dropout_train, dropout_test, method, epsilon, 
            subset, proportion, repeat, len_trial, n_trial, boundary, n_channel]
    npz = args2npz(dataset, 'bayes', args)
    return npz


def get_npz(mode: str, *args):
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


if __name__ == '__main__':
    # mkdir_dataset()
    backup_script()
    pass
