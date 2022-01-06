#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:28:54 2021

@author: chenxiyuan

prepare data for training and testing
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import (ToTensor, Compose, Normalize, 
                                    RandomHorizontalFlip, ToPILImage)

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB

from torchaudio.datasets.speechcommands import SPEECHCOMMANDS

from utils import (get_npz, get_npz_output, get_npz_attack, get_npz_likelihood, 
                   get_npz_bayes, get_npz_cumsum, mkdir_dataset)


# available name of dataset
DATASET = ['mnist', 'cifar10', 'imdb', 'speechcommands']
DATASET_VISION = ['mnist', 'cifar10']
DATASET_TEXT = ['imdb']
DATASET_AUDIO = ['speechcommands']
for i in DATASET:
    mkdir_dataset(i)
# directory containing dataset files
DATASET_DIRECTORY = {'mnist': 'data', 'cifar10': 'data', 'imdb': 'data',
                     'speechcommands': 'data'}
SPEECHCOMMANDS_LABEL = ["zero", "one", "two", "three", "four", "five", "six", 
                        "seven", "eight", "nine", 
                        "yes", "no", "up", "down", "left", "right", "on", "off", 
                        "stop", "go", "backward", "forward", "follow", "learn",
                        "bed", "bird", "cat", "dog", "happy", "house", "marvin", 
                        "sheila", "tree", "wow", "visual"]
SPEECHCOMMANDS_SAMPLINGRATE = 16000


class TextToTensor:
    """
    Tokenize and vectorize strings.

    Parameters
    ----------
    tokenizer : str, optional
        name of tokenizer. The default is 'basic_english'.
    glove_name : str, optional
        version of glove word2vec. The default is '6B'.
    glove_dim : int, optional
        dims of converted vectors. The default is 300.
    max_len : int, optional
        number of tokens, strings will be padded or trunated if shorter or 
        longer than this number. The default is 400.

    Returns
    -------
    token vectors.

    """
    def __init__(self,
                 tokenizer: str = 'basic_english',
                 glove_name: str = '6B',
                 glove_dim: int = 300,
                 max_len: int = 400):

        self.tokenizer = get_tokenizer(tokenizer)
        self.vocab = GloVe(name=glove_name, dim=glove_dim)
        self.max_len = max_len

    def __call__(self, text):
        x = self.tokenizer(text)
        len_x = len(x)
        x = x[:self.max_len] if len_x >= self.max_len else x + [''] * (self.max_len-len_x)
        x = self.vocab.get_vecs_by_tokens(x)
        return x


class CustomDataset(Dataset):
    """
    Instantiate a pytorch dataset with given x and y.

    Parameters
    ----------
    x : np.ndarray, shape as [n_sample, *input_shape]
        inputs for model.
    y : np.ndarray, shape as [n_sample]
        true labels in indexs.
    transform_x : None, optional
        transformation of x. The default is None.

    Returns
    -------
    None.

    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 transform_x: None = None):
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
        self.transform_x = transform_x

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform_x is not None:
            x = self.transform_x(x)
        y = self.targets[idx]

        return x, y


def get_n_sample_dataset(dataset: str, torch_dataset: torch.utils.data.Dataset):
    """
    Return number of samples in given dataset.

    Parameters
    ----------
    dataset : str
        name of dataset.
    torch_dataset : torch.utils.data.Dataset
        torch dataset.

    Returns
    -------
    n_sample : int
        number of samples.

    """
    if dataset in DATASET_VISION:
        n_sample = torch_dataset.data.shape[0]
    elif dataset in DATASET_TEXT:
        n_sample = torch_dataset.num_lines
    return n_sample


def get_dataset(dataset: str = 'mnist',
                subset: str = 'test',
                proportion: float = 1,
                validation_split: float = 0.2,
                transform: bool = True,
                phase: str = None):
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
    transform : bool, optional
        whether apply transformation to inputs before feeding to model. 
        The default is True.
    phase : str, optional
        ['training' | 'attacking' | None], for cifar10+renset only. 
        The default is None.

    Returns
    -------
    dataset_return : CustomDataset
        pytorch dataset.

    """
    dataset_directory = DATASET_DIRECTORY[dataset]
    generator = torch.Generator().manual_seed(0)
    
    if dataset == 'speechcommands':
        subset_dict = {'train': 'training', 'val': 'validation', 'test': 'testing'}
        ds = SPEECHCOMMANDS(dataset_directory, subset=subset_dict[subset])
        n_sample = int(ds.__len__() * proportion)
        subset_return, _ = random_split(ds, [n_sample, ds.__len__()-n_sample], 
                                        generator=generator)
        
        x = []
        y = []
        length = 16000
        for i in range(n_sample):
            # waveform, sample_rate, label, speaker_id, utterance_number
            item = subset_return.__getitem__(i)
            x_i = item[0]
            # pad waveform if shorter than one second
            x_len = x_i.shape[-1]
            if x_len < length:
                x_i = torch.cat([x_i, torch.zeros([1, length-x_len])], dim=-1)
            # cut waveform if longer than one second
            elif x_len > length:
                x_i = x_i[:, :length]
                
            x.append(x_i[0])
            y.append(SPEECHCOMMANDS_LABEL.index(item[2]))
            
        x = torch.stack(x)
        dataset_return = CustomDataset(x, y)
        return dataset_return
          
    
    if subset in ['train', 'val']:
        if dataset == 'mnist':
            train_data = MNIST(dataset_directory, True, ToTensor(), download=True)
        elif dataset == 'cifar10':
            train_data = CIFAR10(dataset_directory, True, ToTensor(), download=True)
        elif dataset == 'imdb':
            train_data = IMDB(root=dataset_directory, split='train')
        # split full training set into real training set and validation set
        n_fulltrain = get_n_sample_dataset(dataset, train_data)
        n_val = int(n_fulltrain * validation_split)
        n_train = n_fulltrain - n_val
        lengths = [int(n_train * proportion), n_train - int(n_train * proportion),
                   int(n_val * proportion), n_val - int(n_val * proportion)]
        train_subset, _,  val_subset, _ = random_split(train_data, lengths,
                                                       generator=generator)
        subset_return = train_subset if subset == 'train' else val_subset
    elif subset == 'test':
        if dataset == 'mnist':
            test_data = MNIST(dataset_directory, False, ToTensor(), download=True)
        elif dataset == 'cifar10':
            test_data = CIFAR10(dataset_directory, False, ToTensor(), download=True)
        elif dataset == 'imdb':
            test_data = IMDB(root=dataset_directory, split='test')
        n_test = get_n_sample_dataset(dataset, test_data)
        lengths = [int(n_test * proportion), n_test - int(n_test * proportion)]
        subset_return, _ = random_split(test_data, lengths, generator)

    if dataset in DATASET_VISION:
        if dataset == 'mnist':
            x = subset_return.dataset.data.numpy()[subset_return.indices]
            y = subset_return.dataset.targets.numpy()[subset_return.indices]
            transform_x = ToTensor() if transform else None
        elif dataset == 'cifar10':
            x = subset_return.dataset.data[subset_return.indices]
            y = np.array(subset_return.dataset.targets)[subset_return.indices]
            # these transform only suits resnet
            if transform:
                normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                if phase == 'training':
                    transform_x = Compose([ToPILImage(),
                                           RandomHorizontalFlip(), 
                                           ToTensor(), 
                                           normalize])
                elif phase == 'attacking':
                    transform_x = Compose([ToPILImage(), ToTensor()])
                else:
                    transform_x = Compose([ToPILImage(), ToTensor(), normalize])
            else:
                transform_x = None
        

    elif dataset in DATASET_TEXT:
        idx_subset = subset_return.indices
        x, y = [], []
        for idx, (iy, ix) in enumerate(subset_return.dataset):
            if idx in idx_subset:
                x.append(ix)
                iy = 0 if iy == 'neg' else 1
                y.append(iy)
        transform_x = TextToTensor() if transform else None

    dataset_return = CustomDataset(x, y, transform_x)

    return dataset_return


def get_loader(dataset: str = 'mnist',
               architecture: str = 'cnn',
               index: int = 0,
               dropout_train: float = 0,
               dropout_test: float = 0,
               method: str = 'clean',
               epsilon: float = 0,
               subset: str = 'test',
               proportion: float = 1,
               batch_size: int = 64,
               num_workers: int = 4,
               shuffle: bool = False, 
               phase: str = None):
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
    num_workers: int, optional
        how many subprocesses to use for data. The default is 4.
    shuffle : bool, optional
        whether return data in random order. The default is False.
    phase : str, optional
        ['training' | 'attacking' | None], for cifar10+renset only. 
        The default is None.
        

    Returns
    -------
    loader : torch.utils.data.DataLoader
        pytorch data loader.

    """

    d = get_dataset(dataset, subset, proportion, phase=phase)
    # load adversarial samples
    if method != 'clean':
        if dataset in DATASET_TEXT:
            transform = TextToTensor()
        elif dataset == 'cifar10':
            normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            transform = Compose([lambda x: torch.tensor(x), normalize])
        else:
            transform = None
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
        d = CustomDataset(x, y, transform)

    loader = DataLoader(d, batch_size, shuffle, num_workers=num_workers)

    return loader


def iter2array(x):
    """
    Convert list or torch.Tensor into numpy.ndarray

    Parameters
    ----------
    x : iterable type 
        list, torch.Tensor or numpy.ndarray.

    Raises
    ------
    Exception
        x belongs to other types.

    Returns
    -------
    x : np.ndarray
        converted x.

    """
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, torch.Tensor):
        x = x.numpy()
    elif isinstance(x, list):
        x = np.array(x)
    else:
        raise Exception("Invalid type: %s"%str(type(x))) 
    
    return x


def get_y(dataset: str = 'mnist',
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
    y : np.ndarray
        labels.

    """
    dataset_torch = get_dataset(dataset, subset, proportion)

    y = dataset_torch.targets
    y = iter2array(y)
    if onehot:
        y = np.eye(np.max(y)+1)[y]

    return y


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
              index: int = 0,
              dropout_train: float = 0,
              dropout_test: float = 0,
              method: str = 'clean',
              epsilon: float = 0,
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
    y = np.repeat(get_y(dataset, subset, proportion), n_trial)

    return trial, y


def get_output_of_category(dataset: str = 'mnist',
                           architecture: str = 'cnn',
                           index: int = 0,
                           dropout_train: float = 0,
                           dropout_test: float = 0,
                           method: str = 'clean',
                           epsilon: float = 0,
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
    y = get_y(dataset, subset, proportion)
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
               n_channel: int,
               likelihood_method: list,
               likelihood_epsilon: list):
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
    likelihood_method: list
        strings of attack methods for likelihood estimation.
    likelihood_epsilon: list
        floats of attack epsilons for likelihood estimation.

    Returns
    -------
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by bayesian inference.
    evidence : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all posterior probabilities through the process of bayesian inference.

    """
    args = [dataset, architecture, index, dropout_train, dropout_test, method, 
            epsilon, subset, proportion, repeat, len_trial, n_trial, boundary, 
            n_channel, likelihood_method, likelihood_epsilon]
    npz = get_npz_bayes(*args)
    with np.load(npz, allow_pickle=True) as data:
        evidence = data['evidence']
        response_time = data['response_time']
        choice = data['choice']

    return response_time, choice, evidence


def load_cumsum(dataset: str,
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
                boundary: float,):
    """
    Return saved Cumsum inference results.

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
    response_time : np.ndarray, shape as [n_sample*n_trial]
        number of timesteps needed to make the decision.
    choice : np.ndarray, shape as [n_sample*n_trial]
        final decisions made by cumsum inference.
    evidence : np.ndarray, shape as [n_sample*n_trial, len_trial, n_choice]
        all summed prediciton through the process of cumsum inference.

    """
    args = [dataset, architecture, index, dropout_train, dropout_test, method, 
            epsilon, subset, proportion, repeat, len_trial, n_trial, boundary]
    npz = get_npz_cumsum(*args)
    with np.load(npz, allow_pickle=True) as data:
        evidence = data['evidence']
        response_time = data['response_time']
        choice = data['choice']

    return response_time, choice, evidence    
    


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
    ds = get_dataset('speechcommands', proportion=0.1)
    # dl = get_loader(dataset='imdb', proportion=0.5)
    # for x, y in dl:
    #     x = x.numpy()
    #     y = y.numpy()
    pass
