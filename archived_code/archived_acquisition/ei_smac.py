# encoding=utf8
import abc
from typing import Any, List, Tuple

import copy

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter
from scipy.stats import norm

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import BaseEPM
from smac.utils.logging import PickableLoggerAdapter



class AbstractAcquisitionFunction(object):
    """Abstract base class for acquisition function.

    Parameters
    ----------
    model : BaseEPM
        Models the objective function.

    Attributes
    ----------
    model
    logger
    """

    def __init__(self, model):
        self.model = model
        self._required_updates = ("model",)  # type: Tuple[str, ...]

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function. As an examples, EI uses it to update the current fmin.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        Parameters
        ----------
        kwargs
        """
        for key in self._required_updates:
            if key not in kwargs:
                raise ValueError(
                    "Acquisition function %s needs to be updated with key %s, but only got "
                    "keys %s." % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def __call__(self, configurations: List[Configuration]) -> np.ndarray:
        """Computes the acquisition value for a given X.

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        X = convert_configurations_to_array(configurations)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the acquisition value for a given point X. This function has to be overwritten
        in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()

class EI(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.
    """

    def __init__(self, model: BaseEPM, par: float = 0.0):
        """Constructor.

        Parameters
        ----------
        model : BaseEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EI, self).__init__(model)
        self.long_name = "Expected Improvement"
        self.par = par
        self.eta = None
        self._required_updates = ("model", "eta")

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

        return f