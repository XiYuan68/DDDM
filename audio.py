#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 23:09:59 2021

@author: chenxiyuan
"""
from time import time

import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from art.estimators.speech_recognition.speech_recognizer import (SpeechRecognizerMixin, 
                                                                 PytorchSpeechRecognizerMixin)
from art.attacks.evasion import ImperceptibleASRPyTorch

from task import (SPEECHCOMMANDS_LABEL, SPEECHCOMMANDS_SAMPLINGRATE, 
                  get_dataset, get_loader)
from model import load_model
from traintest import predict_epoch, DROPOUT
from utils import get_npz


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeechRecognizer(PytorchSpeechRecognizerMixin, 
                       SpeechRecognizerMixin,
                       PyTorchClassifier):
    def __init__(self, 
                 dataset='speechcommands', 
                 architecture: str = 'deepspeech', 
                 index: int = 0, 
                 dropout_train: float = 0, 
                 dropout_test: float = 0,
                 device: torch.device = DEVICE):
        self.device_original = device
        self.model_original = load_model(dataset, architecture, index, 
                                         dropout_train, dropout_test, device)
        self.loss_original = torch.nn.CrossEntropyLoss()
        self.optimizer_original = torch.optim.Adam(self.model_original.parameters())
        self.label_list: list = eval(dataset.upper()+'_LABEL')
        self.sampling_rate: int = eval(dataset.upper()+'_SAMPLINGRATE')
        self.sample_rate = self.sampling_rate
        
        PyTorchClassifier.__init__(self, 
                                   self.model_original, 
                                   input_shape=(self.sampling_rate,),
                                   nb_classes=len(self.label_list),
                                   loss=self.loss_original,
                                   optimizer=self.optimizer_original,
                                   clip_values=(-1, 1))
        
    
    def compute_loss_and_decoded_output(self, 
                                        masked_adv_input: torch.Tensor, 
                                        original_output: np.ndarray, 
                                        **kwargs) -> tuple[torch.Tensor, np.ndarray]:
        # masked_adv_input.dtype == float64
        masked_adv_input = masked_adv_input.type(torch.float32)
        output = self.model_original(masked_adv_input)
        # str --> int
        original_output = [self.label_list.index(i) for i in original_output]
        original_output = torch.Tensor(original_output).type(torch.long).to(self.device_original)
        loss = self.loss_original(output, original_output)
        
        output_argmax = torch.argmax(output, dim=-1)
        label = np.array([self.label_list[i] for i in output_argmax])
        
        return loss, label

    
    def to_training_mode(self) -> None:
        """
        Put the estimator in the training mode.
        """
        self.model.train()


    def sample_rate(self) -> int:
        """
        Get the sampling rate.

        :return: The audio sampling rate.
        """
        return self.sampling_rate
    

def attack_model_artasr(dataset: str = 'speechcommands', 
                        architecture: str = 'deepspeech',
                        index: int = 0, 
                        dropout_train: float = 0, 
                        dropout_test: float = 0, 
                        method: str = 'imperceptible', 
                        epsilon: float = 0.05, 
                        subset: str = 'test', 
                        proportion: float = 0.1, 
                        batch_size: int = 256, 
                        save: bool = True,
                        device: torch.device = DEVICE,
                        num_workers: int = 4):
    print('Attacking', dataset, architecture, index, dropout_train, dropout_test, 
          method, epsilon, subset, proportion)

    recognizer = SpeechRecognizer(dataset, architecture, index, dropout_train,
                                  dropout_test, device)

    if method == 'imperceptible':
        attacker = ImperceptibleASRPyTorch(recognizer, epsilon, max_iter_1=30,
                                           max_iter_2=100, batch_size=batch_size)
    else:
        raise Exception('Invalid Attack Method: %s'%method)
    
    dataset_torch = get_dataset(dataset, subset, proportion)
    x = dataset_torch.data.numpy()
    y = np.array(dataset_torch.targets)
    # random target label for each audio clip
    y_target = np.random.randint(1, np.max(y), y.shape[0])
    y_target = (y_target + y) % (np.max(y) + 1)
    label_target = np.array([recognizer.label_list[i] for i in y_target])

    x_attacked = attacker.generate(x, label_target)
    
    if save:
        npz = get_npz('attack', dataset, architecture, index, dropout_train, 
                      dropout_test, method, epsilon, subset, proportion)
        np.savez_compressed(npz, x=x_attacked, acc=[None])

        loader = get_loader(dataset, architecture, index, dropout_train,
                            dropout_test, method, epsilon, subset, proportion,
                            batch_size)
        acc = predict_epoch(dataset, architecture, recognizer.model, loader,
                            device=device)[-1]
        print('Acc: %f'%acc)
        np.savez_compressed(npz, x=x_attacked, acc=[acc])
    
    return x_attacked
    

if __name__ == '__main__':
    # for subset='train', proportion = 0.1, it takes 1569 mins to attack all models
    time0 = time()
    
    for i in DROPOUT:
        for j in DROPOUT:
            attack_model_artasr(dropout_train=i, dropout_test=j, subset='train')
            print('Cost %.2f mins'%((time() - time0) / 60))