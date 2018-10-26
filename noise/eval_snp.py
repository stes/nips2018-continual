""" Datasets and Augmentation strategies
"""

from torchvision import transforms

import h5py

import os
import os.path as osp
import sys
import argparse

import numpy as np

import torch
from salad import models

import solver  

from salad.utils.evaluate import evaluate
import datasets_snp as datasets

def build_parser():

    parser = argparse.ArgumentParser(description='Multidomain Noise')
    parser.add_argument('--checkpoint', default="", help="Checkpoint path")    
    parser.add_argument('--source', type=str, choices=['noise', 'clean'], help="Batch size of Source")

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()
    
    if args.source == 'noise':
        noisemodels = datasets.noise2clean()
    else:
        noisemodels = datasets.clean2noise()

    data, loader  = datasets.get_dataset(noisemodels, 128, num_workers = 4, which='test', shuffle=False)

    results = []

    savefile = osp.basename(args.checkpoint)[:-4] + '-snp-' + args.source + '.hdf5'
    print(savefile)
    
    if osp.exists(savefile):
        print('deleting previous file')
        os.unlink(savefile)
        
    root = [args.checkpoint]
    
    with h5py.File(savefile) as ds:
        for i in range(len(data)):
            for j in range(len(data)):
                l,y,f = evaluate(root, data[i], j)
                results.append([i,j,l,y,f])
                print(i,j,(l == y.argmax(axis=-1)).mean())

                key = '{}_{}'.format(i,j)

                grp = ds.require_group(key)

                grp['lbl']   = l
                grp['pred']  = y
                grp['feats'] = f