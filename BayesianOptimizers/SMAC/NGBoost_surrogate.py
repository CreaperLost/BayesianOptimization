import numpy as np

import numpy  as np
import pandas as pd
import torch
from copy import deepcopy
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import numpy as np
import torch
from pybnn.bohamiann import Bohamiann
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
from ngboost import NGBoost,NGBRegressor


class  NGBoost_Surrogate(BaseModel):
    def __init__(self, config_space, rng=None):
        
        self.config_space = config_space

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.ngboost = NGBRegressor(n_estimators=100,random_state=rng,verbose=False)        
        self.normalize_y = True
        super(NGBoost_Surrogate, self).__init__()


    def train(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        X = self._impute_inactive(X)
 
        tmp_y = y.copy()

        
        y = tmp_y


        if self.normalize_y:
            y = self._normalize_y(y)
        y = y.flatten()
        

        self.ngboost.fit(X,y)
        
        


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

        #Returns dist object
        dist = self.ngboost.pred_dist(X)
        #dist.loc is the mean, dist.var is the variance

        mean,var = dist.loc , dist.var 
        
        if self.normalize_y:
            mean, var = self._untransform_y(mean, var)

        return mean.reshape([-1,1]), var.reshape([-1,1]) 


