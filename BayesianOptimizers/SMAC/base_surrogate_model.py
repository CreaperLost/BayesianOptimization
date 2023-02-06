import abc
import numpy as np
from typing import Dict, List, Optional, Tuple
from typing import List, Optional, Tuple, Union

import numpy as np

from smac.configspace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Abstract base class for all models
        """
        self.conditional = dict()  # type: Dict[int, bool]
        self.impute_values = dict()  # type: Dict[int, float]
    
    @abc.abstractmethod
    def train(self, X, y):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        pass

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx, hp in enumerate(self.config_space.get_hyperparameters()):
            if idx not in self.conditional:
                parents = self.config_space.get_parents_of(hp.name)
                if len(parents) == 0:
                    self.conditional[idx] = False
                else:
                    self.conditional[idx] = True
                    if isinstance(hp, CategoricalHyperparameter):
                        self.impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        self.impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self.impute_values[idx] = 1
                    else:
                        raise ValueError

            if self.conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = self.impute_values[idx]

        return X


    @abc.abstractmethod
    def predict(self, X_test):
        """
        Predicts for a given set of test data points the mean and variance of its target values

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N Test data points with input dimensions D

        Returns
        ----------
        mean: ndarray (N,)
            Predictive mean of the test data points
        var: ndarray (N,)
            Predictive variance of the test data points
        """
        pass

    def _check_shapes_train(func):
        def func_wrapper(self, X, y, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)
        return func_wrapper

    def _check_shapes_predict(func):
        def func_wrapper(self, X, *args, **kwargs):
            assert len(X.shape) == 2
            return func(self, X, *args, **kwargs)

        return func_wrapper


    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean unit standard deviation.

        Parameters
        ----------
        y : np.ndarray
            Targets for the Gaussian process

        Returns
        -------
        np.ndarray
        """
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_


    def _untransform_y(
        self,
        y: np.ndarray,
        var: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform zeromean unit standard deviation data into the regular space.

        This function should be used after a prediction with the Gaussian process which was
        trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on Tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_**2
            return y, var  # type: ignore
        return y
