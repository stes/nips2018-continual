""" Solver used for the experiments here
"""

from salad.solver import Solver
from salad.layers import MeanAccuracyScore

import torch
from torch import nn


class MultidomainLoss(object):

    def __init__(self, solver, domain):

        self.domain     = domain
        self.solver = solver

    def __call__(self, batch):
        losses = {}

        (x, y)  = batch[self.domain]

        _ , y_ = self.solver.model(x, self.domain)

        # print(y_.size(), y.size(), x.size())

        if not self.solver.multiclass:
            y_ = y_.squeeze(1)
            y  = y.float()
        
        losses['CE_{}'.format(self.domain)] = (y_, y)
        losses['ACC_{}'.format(self.domain)] = (y_, y)

        return losses


class MultidomainBCESolver(Solver):

    def __init__(self, model, dataset, learningrate, multiclass = True,
                 loss_weights = None, *args, **kwargs):

        super(MultidomainBCESolver, self).__init__(dataset=dataset,
                                                   *args, **kwargs)

        self.model      = model
        self.multiclass = multiclass
        self.n_domains  = model.n_domains
        if loss_weights is not None:
            weights = self.cuda(torch.tensor(loss_weights).float())
        else:
            weights = None

        self.register_model(self.model, "domain")

        for d in range(self.n_domains):

            loss_func = nn.CrossEntropyLoss(weight=weights) if multiclass else nn.BCEWithLogitsLoss()
            self.register_loss( loss_func, weight = 1,  name   = 'CE_{}'.format(d), display=False)
            self.register_loss( MeanAccuracyScore(),  weight = None, name   = 'ACC_{}'.format(d))

            opt = torch.optim.Adam(self.model.parameters(d, yield_shared = (d==0)),
                                        lr=kwargs.get('learningrate', learningrate),
                                        amsgrad=True)

            self.register_optimizer(opt, MultidomainLoss(self, domain=d))

    def __repr__(self):
        pass
    
class MultidomainJointSharedBCESolver(Solver):

    def __init__(self, model, dataset, learningrate, multiclass = True,
                 loss_weights = None, *args, **kwargs):

        super(MultidomainBCESolver, self).__init__(dataset=dataset,
                                                   *args, **kwargs)

        self.model      = model
        self.multiclass = multiclass
        self.n_domains  = model.n_domains
        if loss_weights is not None:
            weights = self.cuda(torch.tensor(loss_weights).float())
        else:
            weights = None

        self.register_model(self.model, "domain")

        for d in range(self.n_domains):

            loss_func = nn.CrossEntropyLoss(weight=weights) if multiclass else nn.BCEWithLogitsLoss()
            self.register_loss( loss_func, weight = 1,  name   = 'CE_{}'.format(d), display=False)
            self.register_loss( MeanAccuracyScore(),  weight = None, name   = 'ACC_{}'.format(d))

            if d == 0:
                params = set(self.model.parameters(0, yield_shared = True))
                for dd in range(self.n_domains):
                    params = params.union(set(self.model.parameters(dd, yield_shared = True)))
            else:
                params = self.model.paramters(d, yield_shared = False)
            
            opt = torch.optim.Adam(list(params),
                                        lr=kwargs.get('learningrate', learningrate),
                                        amsgrad=True)

            self.register_optimizer(opt, MultidomainLoss(self, domain=d))

    def __repr__(self):
        pass