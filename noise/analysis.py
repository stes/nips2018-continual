""" Helper functions for result analysis once a model is trained.
"""


import h5py
import glob

import torch
from torch import nn
import numpy as np

import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import itertools
import numpy as np
import torch

def plot_overview(ACC, x, noise_var, titles):

    fig = plt.figure(figsize=(10,10))
    
    sns.set_context('poster')

    


    grid = GridSpec(11,11)

    ax_from = plt.subplot(grid[1:-1  , :1])
    ax_to   = plt.subplot(grid[-1: , 1:])
    ax_img  = plt.subplot(grid[1:-1 ,1:])

    from_img = x.reshape(-1,28)
    to_img   = x.transpose((1,0,2)).reshape(28,-1)

    ax_from.imshow(from_img,aspect='auto',cmap='gray')
    ax_to.imshow(to_img,aspect='auto',cmap='gray')

    sns.heatmap(ACC.T*100, cmap='Blues', annot=True, fmt='.0f', vmin=25, vmax=100, ax=ax_img, cbar=None,
                xticklabels=[], yticklabels=[], linewidths = 1
               )

    step = 28

    ax_to.set_xticks(range(step//2,step*len(titles),step))
    ax_to.set_yticks([])
    ax_from.set_xticks([])
    ax_from.set_yticks(range(step//2,step*len(titles),step))
    ax_to.set_xticklabels([])
    ax_to.set_xticks([])
    ax_from.set_yticklabels(titles)

    sns.despine(top=True, bottom=True, left=True, ax=ax_from)
    sns.despine(left=True, bottom=True, top=True, ax=ax_to)

    ax_from.set_ylabel("Adapt Domain")
    ax_to.set_xlabel("Test Domain")
    
    return fig, ax_img


def get_transform(bn_layer):
    
    mu    = bn_layer.running_mean
    var   = bn_layer.running_var
    gamma = bn_layer.weight
    beta  = bn_layer.bias
    
    return mu, var, gamma, beta

def compute_linear(params):

    mu, var, gamma, beta = params

    inv_std = (1e-5 + var)**(-.5)

    b = beta - (mu * gamma) * inv_std 
    m = gamma * inv_std

    return m, b


def load_params(fnames, N = 9): 
    param_collection = [{}, {}, {}, {}]
    
    n_models = len(fnames)

    for n, fname in enumerate(fnames):
        model = torch.load(fname,map_location=lambda storage, loc: storage)
        
        print(n, len(param_collection))

        for i in range(N):

            for j, layer in enumerate(model.conditional_layers):
                
                vals = get_transform(layer.layers[i])
                
                for val, param_dict in zip(vals, param_collection): 
                    
                    val = val.data.detach().cpu().numpy()

                    p             = param_dict.get(j, np.zeros((10,n_models) + val.shape))
                    p[i,n]        = val
                    param_dict[j] = p

    P = []
    for p in param_collection:
        params = [p[i] for i in range(len(p))]
        P.append(np.concatenate(params, axis=-1))
    P = np.stack(P, axis=0)[:,:9]

    return P


def load_file(fname, n_domains, revert=False):

    l_shape = (n_domains,n_domains,26032)
    y_shape = (n_domains,n_domains,26032,10)

    lbl  = np.zeros(l_shape)
    pred = np.zeros(y_shape)

    with h5py.File(fname, 'r') as ds:

        for i,j in itertools.product(range(n_domains), range(n_domains)):

            key       = '{}_{}'.format(i,j)

            grp       = ds.require_group(key)

            lbl[i,j]  = grp['lbl'][...]
            pred[i,j] = grp['pred'][...]

    H = np.equal(lbl, pred.argmax(axis=-1))
    ACC = H.mean(axis=-1)

    if revert:
        ACC = ACC[::-1, ::-1]
    
    return ACC

def plot_reg(x,y,ax = None):
    ax = ax or plt.gca()

    sns.regplot(x*100, y*180/np.pi, ax=ax)
    sns.despine(ax = ax)

    ax.set_xlabel('Accuracy [%]')
    ax.set_ylabel('Angle [°]')
    
    return ax

def get_corr(ACC, a, filt=np.tril):

    x = filt(ACC, 1)
    y = filt(a, 1)
    idc = filt(np.ones((len(x), len(y)))) > 0

    x = x[idc]
    y = y[idc]
    x = x[y>0.001]
    y = y[y>0.001]
    
    return x, y

def compute_angle(m):
    a    = m.dot(m.T)
    norm = (m**2).sum(axis=-1)**.5

    cos_a = a / (norm[np.newaxis,:] * norm[:,np.newaxis])
    cos_a = np.clip(cos_a,-1,1)
    a = np.arccos(cos_a)
    
    return a

def transfer_plot(a, ACC, ax_angle, ax_acc, noise_var):

    cax = ax_angle.imshow(a*180/np.pi, cmap='Blues')
    cb = plt.colorbar(ax=ax_angle, mappable=cax)
    cb.outline.set_edgecolor(None)
    ax_angle.set_title("Angle (°)")
    
    ax_angle.set_yticklabels(noise_var)
    ax_angle.set_yticks([]) #np.arange(9),minor=True)

    cax = ax_acc.imshow(ACC.T, cmap='Blues_r')
    cb = plt.colorbar(ax=ax_acc, mappable=cax)
    cb.outline.set_edgecolor(None)
    ax_acc.set_title("Transfer Accuracy")
    
    ax_angle.axis('off')
    ax_acc.axis('off')