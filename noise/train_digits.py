""" Datasets and Augmentation strategies
"""

import os
import os.path as osp
import sys
import argparse

import numpy as np

import torch

from salad import models
from salad.datasets import DigitsLoader

import solver  
    
def build_parser():

    parser = argparse.ArgumentParser(description='Multidomain Noise')

    # General setup
    parser.add_argument('--gpu', default=0, help='Specify GPU', type=int)
    parser.add_argument('--cpu', action='store_true', help="Use CPU Training")
    parser.add_argument('--log', default="./log/digits", help="Log directory. Will be created if non-existing")
    parser.add_argument('--epochs', default="100", help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
    parser.add_argument('--checkpoint', default="", help="Checkpoint path")
    parser.add_argument('--learningrate', default=3e-4, type=float, help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
    parser.add_argument('--dryrun', action='store_true', help="Perform a test run, without actually training a network. Usefule for debugging.")
    parser.add_argument('--batchsize', default=64, type=int, help="Batch size of Source")
    
    parser.add_argument('--source', type=str, required = True, choices=['mnist', 'usps', 'synth', 'svhn'], help="Batch size of Source")


    return parser


if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()
    
    datasets = ['mnist', 'usps', 'synth', 'svhn']
    datasets.pop(datasets.index(args.source))
    datasets = [args.source] + datasets
    print(datasets)

    loader   = DigitsLoader('/tmp/data', datasets, download=True, shuffle=True, batch_size=64,
                                                   normalize=True, num_workers=4)
    model   = models.ConditionalModel(4)

    # Initialize the solver for this experiment
    experiment = solver.MultidomainBCESolver(model, loader,
                               n_epochs=args.epochs,
                               savedir=osp.join(args.log, args.source),
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)
    
    experiment.optimize()
