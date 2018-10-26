""" Noise Adaptation between Gaussian and Salt and Pepper Noise Model

Stores all results in ``./log/multitask/mixed``
"""

from torchvision import datasets, transforms

import os
import os.path as osp
import sys
import argparse

import numpy as np

import torch

from salad.models.digits.fan import ConditionalModel

import datasets_sp as datasets
import solver  
    
def build_parser():

    parser = argparse.ArgumentParser(description='Multidomain Noise')

    # General setup
    parser.add_argument('--gpu', default=0, help='Specify GPU', type=int)
    parser.add_argument('--cpu', action='store_true', help="Use CPU Training")
    parser.add_argument('--log', default="./log/multitask/mixed", help="Log directory. Will be created if non-existing")
    parser.add_argument('--epochs', default="100", help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
    parser.add_argument('--checkpoint', default="", help="Checkpoint path")
    parser.add_argument('--learningrate', default=3e-4, type=float, help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
    parser.add_argument('--dryrun', action='store_true', help="Perform a test run, without actually training a network. Usefule for debugging.")
    parser.add_argument('--batchsize', default=64, type=int, help="Batch size of Source")
    
    parser.add_argument('--source', type=str, choices=['white', 'snp'], help="Batch size of Source")
    parser.add_argument('--twodomains', action='store_true', help="Specify for smaller version of the experiment")


    return parser


if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()

    if args.source == 'white':
        noisemodels = datasets.gauss2snp(args.twodomains)
    else:
        noisemodels = datasets.snp2gauss(args.twodomains)

    noisemodels = noisemodels

    _, loader = datasets.get_dataset(noisemodels, args.batchsize, num_workers = 4)
    model     = ConditionalModel(len(noisemodels))

    # Initialize the solver for this experiment
    experiment = solver.MultidomainBCESolver(model, loader,
                               n_epochs=args.epochs,
                               savedir=osp.join(args.log, args.source),
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)
    
    experiment.optimize()