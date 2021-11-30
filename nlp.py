#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:35:25 2021

@author: chenxiyuan

attack text classifier
"""
from time import time

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer

from textattack.models.wrappers import ModelWrapper
from textattack.datasets import Dataset
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018 as textbugger
from textattack import Attacker, AttackArgs

from task import get_dataset, TextToTensor
from model import load_model
from utils import get_dir, get_npz_attack


def get_txt_textattacklog(dataset: str = 'imdb', 
                          architecture: str = 'lstm',
                          index: int = 0, 
                          dropout_train: float = 0, 
                          dropout_test: float = 0, 
                          method: str = 'textbugger', 
                          subset: str = 'test', 
                          proportion: float = 1):
    """
    Return `.txt` file path that saves TextAttack results.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'imdb'.
    architecture : str, optional
        architecture of the neural network. The default is 'lstm'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_train : float, optional
        dropout rate at training phase. The default is 0.
    dropout_test : float, optional
        dropout rate at test phase. The default is 0.
    method : str, optional
        adversarial attack method. The default is 'textbugger'.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is 1.

    Returns
    -------
    txt : str
        `.txt` file path.

    """
    d = get_dir(dataset, 'others')
    args = [dataset, architecture, index, dropout_train, dropout_test, 
            method, subset, proportion]
    txt = d + '_'.join([str(i) for i in args]) + '.txt'
    return txt


def get_accuracy_fromstring(string: str = ''):
    """
    Return float of accuracy in given string.

    Parameters
    ----------
    string : string, optional
        string in format like 'Original accuracy: 88.24%'. The default is ''.

    Returns
    -------
    number : float
        accuracy in given string.

    """
    number = string.split(' ')[-1][:-2]
    number = float(number) / 100
    return number


def get_truncated_string(string: str, tokenizer: None, max_len: int):
    tokens = tokenizer(string)
    if len(tokens)  > max_len:
        tokens = tokens[:max_len]
    string = ' '.join(tokens[:max_len])
    return string


def get_dataset_textattack(dataset: str = 'imdb',
                           subset: str = 'test',
                           proportion: float = 1,
                           tokenizer: str = 'basic_english', 
                           max_len: int = 400):
    """
    Instantiate and return a textattack.datasets.Dataset for textattack.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'imdb'.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is PROPORTION.

    Returns
    -------
    dataset_textattack : textattack.datasets.Dataset
        dataset for textattack.

    """
    dataset_torch = get_dataset(dataset, subset, proportion, transform=False)
    tokenizer = get_tokenizer(tokenizer)
    data = [get_truncated_string(i, tokenizer, max_len) for i in dataset_torch.data]
    xy_pair = [(str(x), y) for x, y in zip(data, dataset_torch.targets)]
    dataset_textattack = Dataset(xy_pair)
    return dataset_textattack


def get_str_adversarial(lines: list, lines_clean: list):
    """
    Return clean text, adversarial text and bool array of adversarial samples
    in log files.

    lines in log txts are typically like this:

        
    -------------- Result 1 ---------------------------------------------
    [[0 (100%)]] --> [[[FAILED]]]
    
    from livesey solntze wpd sgi com 
    ---------------- Result 2 ---------------------------------------------
    [[0 (100%)]] --> [[9 (51%)]]
    
    rom nosubdomain nodomain brian cash [[subject]] re [[free]] 
    
    from nosubdomain nodomain brian cash [[matter]] re [[innocent]]
    ------------------- Result 3 ---------------------------------------------
    [[0 (100%)]] --> [[9 (50%)]]
    
    .
    .
    .


    Parameters
    ----------
    lines : list
        strings in log files.
    lines_clean : list
        strings from torchtext.


    Returns
    -------
    clean_string : list
        clean strings.
    attacked_string : list
        adversarial text.
    adversarial : list
        whether adversarial sample is generated.

    """
    
    def remove_bracket(string: str):
        """
        remove '[[]]' around attacked words because torchtext tokenizer
        will keep them and make the '[[word]]' vectorized as zero vectors

        Parameters
        ----------
        string : str
            DESCRIPTION.

        Returns
        -------
        string : TYPE
            DESCRIPTION.

        """
        string = string.replace('[[', '')
        string = string.replace(']]', '')
        return string
    
    unattacked_string = []
    attacked_string = []
    for idx, i in enumerate(lines):
        if 'Result' not in i:
            continue
        else:
            unattacked_string.append(remove_bracket(lines[idx+3]))
            idx_attacked = idx+3 if 'FAILED' in lines[idx+1] else idx+5
            attacked_string.append(remove_bracket(lines[idx_attacked]))

    # the order of unattacked strings in log file could be in different from
    # ones loaded from torchtext
    idx_correct = [unattacked_string.index(i) for i in lines_clean]
    attacked_string = [attacked_string[i] for i in idx_correct]

    return attacked_string


class TextModelWrapper(ModelWrapper):
    """
    ModelWrapper for textattack.

    Parameters
    ----------
    model : torch.nn.Module
        classifier.
    device : str
        device of torch model.

    """
    def __init__(self, model: torch.nn.Module, device: str):
        self.device = torch.device(device)
        self.model = model
        self.transform = TextToTensor()
    def __call__(self, text_input_list: list):
        x = torch.stack([self.transform(i) for i in text_input_list], dim=0)
        x = x.to(self.device)
        y = self.model(x).cpu().detach().numpy()
        return y


def attack_model_textattack(dataset: str = 'imdb', 
                            architecture: str = 'lstm',
                            index: int = 0, 
                            dropout_train: float = 0, 
                            dropout_test: float = 0, 
                            method: str = 'textbugger', 
                            subset: str = 'test', 
                            proportion: float = 1, 
                            batch_size: int = 1, 
                            save: bool = True,
                            device: str = 'cuda',
                            multigpu: bool = False):
    """
    Attack text classifier with TextAtack.

    Parameters
    ----------
    dataset : str, optional
        name of dataset. The default is 'imdb'.
    architecture : str, optional
        architecture of the neural network. The default is 'lstm'.
    index : int, optional
        index of data to return. The default is 0.
    dropout_train : float, optional
        dropout rate at training phase. The default is 0.
    dropout_test : float, optional
        dropout rate at test phase. The default is 0.
    method : str, optional
        adversarial attack method. The default is 'textbugger'.
    subset : str, optional
        subset of dataset. The default is 'test'.
    proportion : float, optional
        proportion of subset data. The default is 1.
    batch_size : int, optional
        number of parallel attacks. The default is 2.
    save : bool, optional
        whether save the result. The default is True.
    device : str, optional
        device of torch model. The default is 'cuda'.
    multigpu : bool, optional
        whether perform Data Parallelism on multiplt GPUs. The default is False.

    Returns
    -------
    x_attacked : numpy.ndarray
        adversarial(if attack successfuly)/clean strings.
    acc : float
        accuracy after attacks.

    """
    epsilon = 0
    print('Attacking', dataset, architecture, index, dropout_train, dropout_test, 
          method, epsilon, subset, proportion)
    model = load_model(dataset, architecture, index, dropout_train, dropout_test,
                        device, multigpu)
    wrapper = TextModelWrapper(model, device)
    attack = eval(method).build(wrapper)
    dataset_ta = get_dataset_textattack(dataset, subset, proportion)
    txt = get_txt_textattacklog(dataset, architecture, index, dropout_train, 
                                dropout_test, method, subset, proportion) if save else None
    attack_args = AttackArgs(num_examples=-1, disable_stdout=True, 
                             log_to_txt=txt, parallel=batch_size>1, 
                             num_workers_per_device=batch_size)
    attacker = Attacker(attack, dataset_ta, attack_args)
    attacker.attack_dataset()
    
    if save:
        with open(txt) as f:
            lines = f.readlines()
        lines = [i[:-1] for i in lines]
        acc = get_accuracy_fromstring(lines[-5])
        lines_clean = [i[0] for i in dataset_ta._dataset]
        x_attacked = get_str_adversarial(lines, lines_clean)
        npz = get_npz_attack(dataset, architecture, index, dropout_train, 
                              dropout_test, method, epsilon, subset, proportion)
        np.savez_compressed(npz, x=x_attacked, acc=[acc])
        return x_attacked
    


if __name__ == '__main__':
    pass
    time0 = time()

    # attack_model_textattack(proportion=0.01)
    
    # for proportion=0.01, it takes 713 mins to finish
    for i in [0, 0.2, 0.4, 0.6, 0.8]:
        for j in [0, 0.2, 0.4, 0.6, 0.8]:
            for k in ['train', 'test']:
                attack_model_textattack(dropout_train=i, dropout_test=j, 
                                        subset=k, proportion=0.1)
    
    print('Cost %.2f mins'%((time() - time0) / 60))