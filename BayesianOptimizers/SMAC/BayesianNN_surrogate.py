 # Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import sys
from typing import Optional

import numpy  as np
import pandas as pd
import torch
from copy import deepcopy
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from BayesianOptimizers.SMAC.base_surrogate_model import BaseModel
import numpy as np
import torch
from pybnn.bohamiann import Bohamiann

def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(torch.nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = torch.nn.Parameter(torch.FloatTensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            torch.nn.init.constant_(module.bias, val=np.log(1e-2))
        elif type(module) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            torch.nn.init.constant_(module.bias, val=0.0)

    return torch.nn.Sequential(
        torch.nn.Linear(input_dimensionality, 50), torch.nn.Tanh(),
        torch.nn.Linear(50, 50), torch.nn.Tanh(),
        torch.nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


class WrapperBohamiann(BaseModel):

    def __init__(self, get_net=get_default_network, lr=1e-2, use_double_precision=True, verbose=True):
        """
        Wrapper around pybnn Bohamiann implementation. It automatically adjusts the length by the MCMC chain,
        by performing 100 times more burnin steps than we have data points and sampling ~100 networks weights.

        Parameters
        ----------
        get_net: func
            Architecture specification

        lr: float
           The MCMC step length

        use_double_precision: Boolean
           Use float32 or float64 precision. Note: Using float64 makes the training slower.

        verbose: Boolean
           Determines whether to print pybnn output.
        """

        self.lr = lr
        self.normalize_y = True
        self.verbose = verbose
        self.bnn = Bohamiann(get_network=get_net, use_double_precision=use_double_precision)

    def train(self, X, y, **kwargs):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        X = self._impute_inactive(X)
 
        tmp_y = y.copy()

        y = tmp_y
        if self.normalize_y:
            y = self._normalize_y(y)
        y = y.flatten()


        self.bnn.train(X, y, lr=self.lr,
                       num_burn_in_steps=X.shape[0] * 100,
                       num_steps=X.shape[0] * 100 + 10000, verbose=self.verbose)

    def predict(self, X_test):
        if len(X_test.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X_test.shape))
        
        mean,var =   self.bnn.predict(X_test)
        if self.normalize_y:
            mean, var = self._untransform_y(mean, var)

        return mean.reshape([-1,1]), var.reshape([-1,1]) 

