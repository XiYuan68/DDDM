#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:28:54 2021

@author: chenxiyuan

prepare data for training and testing
"""

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, random_split

from utils import get_npz, get_npz_output, get_npz_attack, get_npz_likelihood, get_npz_bayes


# available name of dataset
DATASET = ['mnist', 'cifar10']


class CustomDataset(Dataset):
    """
    Instantiate a pytorch dataset with given x and y.

    Parameters
    ----------
    x : np.ndarray, shape as [n_sample, *input_shape]
        inputs for model.
    y : np.ndarray, shape as [n_sample]
        true labels in indexs.
    transform : None, optional
        transformation of x. The default is ToTensor().

    Returns
    -------
    None.

    """
    def __init__(self, x: np.ndarray, y: np.ndarray, transform: None = ToTensor()):
        """
        Instantiate a pytorch dataset with given x and y.

        Parameters
        ----------
        x : np.ndarray, shape as [n_sample, *input_shape]
            inputs for model.
        y : np.ndarray, shape as [n_sample]
            true labels in indexs.
        transform : None, optional
            transformation of x. The default is ToTensor().

        Returns
        -------
        None.

        """
        self.data = x
        self.targets = y
        self.transform = transform
    def __len__(self):
        return self.targets.shape[0]
    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        y = self.targets[idx]
        
        return x, y


def get_dataset(dataset: str = 'mnist',
                subset: str = 'test',
                proportion: float = 1, 
                validation_split: float = 0.2,
                transform: None = ToTensor()):
    """
    Return pytorch dataset.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    subset : str, optional
        subset of datset, should be one of [train|val|test]. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is 1.
    validation_split : float, optional
        split of original training set to be the validation set. The default is 0.2.
    transform : None, optional
        transformation of inputs. The default is ToTensor().

    Returns
    -------
    dataset_return : CustomDataset
        pytorch dataset.

    """

    generator = torch.Generator().manual_seed(0)
    if subset in ['train', 'val']:
        if dataset == 'mnist':
            train_data = datasets.MNIST('data', True, ToTensor(), download=True)
        elif dataset == 'cifar10':
            train_data = datasets.CIFAR10('data', True, ToTensor(), download=True)
        # split full training set into real training set and validation set
        n_fulltrain = train_data.data.shape[0]
        n_val = int(n_fulltrain * validation_split)
        n_train = n_fulltrain - n_val
        lengths = [int(n_train * proportion), n_train - int(n_train * proportion),
                   int(n_val * proportion), n_val - int(n_val * proportion)]
        train_subset, _,  val_subset, _ = random_split(train_data, lengths,
                                                       generator=generator)
        subset_return = train_subset if subset=='train' else val_subset
    elif subset == 'test':
        if dataset == 'mnist':
            test_data = datasets.MNIST('data', False, ToTensor(), download=True)
        elif dataset == 'cifar10':
            test_data = datasets.CIFAR10('data', False, ToTensor(), download=True)
        n_test = test_data.data.shape[0]
        lengths = [int(n_test * proportion), n_test - int(n_test * proportion)]
        subset_return, _ = random_split(test_data, lengths, generator)
    
    x = subset_return.dataset.data.numpy()[subset_return.indices]
    y = subset_return.dataset.targets.numpy()[subset_return.indices]
    dataset_return = CustomDataset(x, y, transform)
    
    return dataset_return


def get_loader(dataset: str = 'mnist', 
               architecture: str = 'cnn', 
               index : int = 0, 
               dropout_train: float = 0, 
               dropout_test: float = 0,
               method: str = 'clean', 
               epsilon : float = 0,
               subset: str = 'test', 
               proportion: float = 1,
               batch_size: int = 64, 
               shuffle: bool = False):
    """
    Return pytorch data loader.

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
        proportion of subset data. The default is 1.
    batch_size : int, optional
        batch size. The default is 64.
    shuffle : bool, optional
        whether return data in random order. The default is False.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        pytorch data loader.

    """

    d = get_dataset(dataset, subset, proportion)
    # load generated adversarial samples
    if method != 'clean':
        x = load_attack(dataset, 
                        architecture,
                        index, 
                        dropout_train, 
                        dropout_test,
                        method, 
                        epsilon, 
                        subset, 
                        proportion)
        y = d.targets
        d = CustomDataset(x, y, None)
        
    loader = DataLoader(d, batch_size, shuffle, num_workers=8, )
    
    return loader


def get_tensor(dataset: str = 'mnist', 
               subset: str = 'test', 
               proportion: float = 1, 
               onehot: bool = False):
    """
    Return pytorch tensors of inputs and labels of dataset.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is 1.
    onehot : bool, optional
        whether encode labels in the onehot manner. The default is False.

    Returns
    -------
    x : torch.Tensor
        inputs.
    y : torch.Tensor
        labels.

    """
    dataset_string = dataset
    dataset = get_dataset(dataset, subset, proportion)

    x = dataset.data
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = dataset.targets
    y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
    if dataset_string == 'mnist':
        x = x[:, None]
    if onehot:
        y = torch.eye(torch.max(y)+1)[y]

    return x, y


def get_array(dataset: str = 'mnist', 
              subset: str = 'test', 
              proportion: float = 1, 
              onehot: bool = False):
    """
    Return numpy arrays of inputs and labels of dataset.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'mnist'.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is 1.
    onehot : bool, optional
        whether encode labels in the onehot manner. The default is False.

    Returns
    -------
    x : np.ndarray, shape as [n_sample, *input_shape]
        inputs.
    y : np.ndarray, shape as [n_sample] or [n_sample, n_choice]
        labels.

    """
    x, y = get_tensor(dataset, subset, proportion, onehot)
    x, y = x.numpy(), y.numpy()
    return x, y
    

def load_attack(dataset: str, 
                architecture: str, 
                index: int, 
                dropout_train: float, 
                dropout_test: float, 
                method: str, 
                epsilon: float, 
                subset: str, 
                proportion: float):
    """
    Return saved adversarial samples.

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
    x : np.ndarray, shape as [n_sample, *input_shape]
        adversarial samples.

    """
    npz = get_npz_attack(dataset, architecture, index, dropout_train, dropout_test, 
                         method, epsilon, subset, proportion)
    with np.load(npz, allow_pickle=True) as data:
        x = data['x']
    return x
        

def load_output(dataset: str,
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
    Return saved outputs of neural networks.

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
    output : np.ndarray, shape [repeat, n_sample, n_choice]
        outputs of neural networks.

    """
    npz = get_npz_output(dataset, architecture, index, dropout_train, dropout_test, 
                         method, epsilon, subset, proportion, repeat)
    with np.load(npz, allow_pickle=True) as data:
        output = data['output']
        
    return output


def get_trial(dataset: str = 'mnist', 
              architecture: str = 'cnn', 
              index : int = 0, 
              dropout_train: float = 0, 
              dropout_test: float = 0,
              method: str = 'clean', 
              epsilon : float = 0,
              subset: str = 'test', 
              proportion: float = 1,
              repeat: int = 100, 
              len_trial: int = 25,
              n_trial: int = 10):
    """
    Return trials of neural networks prediction for bayesian inference.

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
        proportion of subset data. The default is 1.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100.
    len_trial : int, optional
        length of each trial. The default is 25.
    n_trial : int, optional
        number of trials for each sample. The default is 10.

    Returns
    -------
    trial : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        trials for decision process.
    y : np.ndarray, shape as [n_sample*n_trial]
        true labels of each trial.

    """

    output = load_output(dataset, 
                         architecture, 
                         index, 
                         dropout_train, 
                         dropout_test, 
                         method, 
                         epsilon, 
                         subset, 
                         proportion, 
                         repeat)
    n_repeat, n_sample, n_choice = output.shape
    idx_timestep = np.random.choice(n_repeat, [n_trial, len_trial])
    trial = np.concatenate(np.moveaxis(output[idx_timestep], -2, 0))
    y = np.repeat(get_array(dataset, subset, proportion)[-1], n_trial)

    return trial, y


def get_output_of_category(dataset: str = 'mnist', 
                           architecture: str = 'cnn', 
                           index : int = 0, 
                           dropout_train: float = 0, 
                           dropout_test: float = 0,
                           method: str = 'clean', 
                           epsilon : float = 0,
                           subset: str = 'test', 
                           proportion: float = 1,
                           repeat: int = 100):
    """
    Return neural network outputs of each category for likelihood acquisition.

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
        proportion of subset data. The default is 1.
    repeat : int, optional
        number of neural network prediction of each sample. The default is 100

    Returns
    -------
    output_of_category : list of np.ndarray, shape as [n_choice][repeat*n_sample, n_choice]
        neural network outputs of each category.

    """
    output = load_output(dataset, 
                         architecture,
                         index, 
                         dropout_train, 
                         dropout_test, 
                         method, 
                         epsilon, 
                         subset, 
                         proportion, 
                         repeat)
    y = get_array(dataset, subset, proportion)[-1]
    output_of_category = []
    for i in range(np.max(y)+1):
        idx = y == i
        output_of_category.append(np.concatenate(output[:, idx]))
    return output_of_category


def load_likelihood(dataset: str, 
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
    Return saved likelihood.

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
    combination : np.ndarray, shape as [n_combination, n_channel]
        all possible combinations of discretized outputs.
    likelihood : np.ndarray, shape as [n_combination, n_choice]
        likelihood of each discretized output when input belongs to certain true label.

    """
    npz = get_npz_likelihood(dataset, architecture, index, dropout_train, dropout_test, 
                             method, epsilon, subset, proportion, repeat, n_channel)
    with np.load(npz, allow_pickle=True) as data:
        likelihood = data['likelihood']
        
    return likelihood


def load_bayes(dataset: str, 
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
    """
    Return saved bayesian inference results.

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
    belief_post : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all posterior probabilities through the process of bayesian inference.
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by bayesian inference.

    """
    npz = get_npz_bayes(dataset, architecture, index, dropout_train, dropout_test, 
                        method, epsilon, subset, proportion, repeat, len_trial, 
                        n_trial, boundary, n_channel)
    with np.load(npz, allow_pickle=True) as data:
        belief_post = data['belief_post']
        response_time = data['response_time']
        choice = data['choice']
    
    return belief_post, response_time, choice


def load_responsetime(dataset: str, 
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
    """
    Return response time in saved bayesian inference results.

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
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.

    """
    response_time = load_bayes(dataset, architecture, index, dropout_train, 
                               dropout_test, method, epsilon, subset, proportion, 
                               repeat, len_trial, n_trial, boundary, n_channel)[1]
    return response_time


def load_acc(mode: str, *args: list):
    """
    Return accuracy of given saved results.

    Parameters
    ----------
    mode : str
        type of data, one of [ attack | bayes ].
    *args : list
        other arguments of desired `.npz` file, depend on `utils.get_npz_mode()`.

    Returns
    -------
    acc : float
        accuracy of given saved results.

    """
    npz = get_npz(mode, *args)
    with np.load(npz, allow_pickle=True) as data:
        acc = data['acc'][0]
    return acc


if __name__ == '__main__':
    pass
