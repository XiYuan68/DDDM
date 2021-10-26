#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:39:04 2021

@author: chenxiyuan

train one model on given GPU, allow command line Arguments
"""

import argparse

from traintest import train_model


parser = argparse.ArgumentParser(description="Train a model on given GPU")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--architecture", type=str, help="architecture of model")
parser.add_argument("--index", type=int, help="index")
parser.add_argument("--dropout", type=float, help="dropout rate")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--epochs", type=int, help="max epochs")
parser.add_argument("--save", type=bool, help="whether save best model")
parser.add_argument("--patience", type=int, help="patience before stop training")
parser.add_argument("--device", type=str, help="pytorch device")
parser.add_argument("--num_workers", type=int, help="number of data loader workers")
args = parser.parse_args()


train_model(args.dataset, 
            args.architecture, 
            args.index, 
            args.dropout, 
            args.batch_size, 
            args.epochs, 
            args.save, 
            args.patience,
            args.device,
            args.num_workers)