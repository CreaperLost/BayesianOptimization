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

class  HEBO_RF(BaseModel):
    def __init__(self, config_space, rng=None):
        
        self.config_space = config_space

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.n_estimators =  30
        self.rf = RandomForestRegressor(n_estimators = self.n_estimators,random_state=rng)

        self.num_out = 1
        self.est_noise = torch.zeros(self.num_out)
        
        self.normalize_y = True


        super(HEBO_RF, self).__init__()


    def train(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        X = self._impute_inactive(X)




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


        if self.normalize_y:
            y = self._normalize_y(y)
        y = y.flatten()
        

        self.rf.fit(X,y)
        
        mse = np.mean((self.rf.predict(X).reshape(-1) - y.reshape(-1))**2).reshape(self.num_out)
        self.est_noise = mse


    def predict(self, X: np.ndarray):
        """Predict means and variances for given X.
        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.
        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
    
        mean = self.rf.predict(X).reshape(-1, 1)
        preds = []
        for estimator in self.rf.estimators_:
            preds.append(estimator.predict(X).reshape([-1,1]))
        var = np.var(np.concatenate(preds, axis=1), axis=1)
        
        var +=  self.est_noise



        if self.normalize_y:
            mean, var = self._untransform_y(mean, var)

        return mean.reshape([-1,1]), var.reshape([-1,1]) 


