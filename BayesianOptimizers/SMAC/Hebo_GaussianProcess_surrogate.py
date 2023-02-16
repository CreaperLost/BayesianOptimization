# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
from sklearn.preprocessing import power_transform

import GPy
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor, FloatTensor, LongTensor
from BayesianOptimizers.SMAC.base_surrogate_model import BaseModel
import logging

class HEBO_GP(BaseModel):
    """
    Input warped GP model implemented using GPy instead of GPyTorch

    Why doing so:
    - Input warped GP
    """
    def __init__(self, config_space, rng=None, ):
        
        self.config_space = config_space

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng


        self.verbose      = False
        self.num_epochs   = 200
        self.warp         = True
        self.num_restarts = 10
        self.normalize_y = True
        self.gp = None

        super(HEBO_GP, self).__init__()


    def train(self, X, y): 
        
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        y = y.reshape(-1,1)        

        tmp_y = y.copy()

        try:
            if y.min() <= 0:
                y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=True)
            else:
                y = power_transform(y / y.std(), method = 'box-cox',standardize=True)
                if y.std() < 0.5:
                    y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=True)
            if y.std() < 0.5:
                raise RuntimeError('Power transformation failed')
        except:    
            #Reset y.
            y = tmp_y
        
        X = self._impute_inactive(X)
        if self.normalize_y:
            y = self._normalize_y(y)


        y = y.reshape(-1,1)

        k1  = GPy.kern.Linear(X.shape[1],   ARD = False)
        k2  = GPy.kern.Matern32(X.shape[1], ARD = True)
        k2.lengthscale = np.std(X, axis = 0).clip(min = 0.02)
        k2.variance    = 0.5
        k2.variance.set_prior(GPy.priors.Gamma(0.5, 1), warning = False)
        #Warped GP.
        kern = k1 + k2
        if not self.warp:
            self.gp = GPy.models.GPRegression(X, y, kern)
        else:
            xmin    = np.zeros(X.shape[1])
            xmax    = np.ones(X.shape[1])
            warp_f  = GPy.util.input_warping_functions.KumarWarping(X, Xmin = xmin, Xmax = xmax)
            self.gp = GPy.models.InputWarpedGP(X, y, kern, warping_function = warp_f)
        self.gp.likelihood.variance.set_prior(GPy.priors.LogGaussian(-4.63, 0.5), warning = False)

        self.gp.optimize_restarts(max_iters = self.num_epochs, verbose = self.verbose, num_restarts = self.num_restarts, robust = False)
        return self

    def predict(self, X):

        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
    
        mean,var = self.gp.predict(X)

        if self.normalize_y:
            mean, var = self._untransform_y(mean, var)

        return mean.reshape([-1,1]), var.reshape([-1,1]) 


    def sample_f(self):
        raise NotImplementedError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def noise(self):
        var_normalized = self.gp.likelihood.variance[0]
        return (var_normalized * self.yscaler.std**2).view(self.num_out)
