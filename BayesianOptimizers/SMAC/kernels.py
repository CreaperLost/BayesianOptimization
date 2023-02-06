from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
from inspect import Signature, signature

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels
from typing import List, Optional, Tuple



import math
import warnings

import numpy as np
import scipy.stats as sps

from BayesianOptimizers.SMAC.utils import VERY_SMALL_NUMBER


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Prior(object):
    """Abstract base class to define the interface for priors of GP hyperparameter.

    This class is adapted from RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    [16.04.2019]: Whenever lnprob or the gradient is computed for a scalar input, we use math.* rather than np.*

    Parameters
    ----------
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(self, rng: np.random.RandomState):
        if rng is None:
            raise ValueError("Argument rng must not be `None`.")
        self.rng = rng

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on a log scale! This method exponentiates theta and calls ``self._lnprob``.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        return self._lnprob(np.exp(theta))

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration on the original scale.

        Returns
        -------
        float
            The log probability of theta
        """
        raise NotImplementedError()

    def sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a log scale. This method calls ``self._sample_from_prior`` and applies a log transformation
        to the obtained values.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)
        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        sample = np.log(self._sample_from_prior(n_samples=n_samples))

        if np.any(~np.isfinite(sample)):
            raise ValueError("Sample %s from prior %s contains infinite values!" % (sample, self))

        return sample

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a original scale.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError()

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        return self._gradient(np.exp(theta))

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in the original space space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        raise NotImplementedError()


class TophatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, rng: np.random.RandomState):
        """Tophat prior as it used in the original spearmint code.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        lower_bound : float
            Lower bound of the prior. In original scale.
        upper_bound : float
            Upper bound of the prior. In original scale.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)
        self.min = lower_bound
        self._log_min = np.log(lower_bound)
        self.max = upper_bound
        self._log_max = np.log(upper_bound)
        if not (self.max > self.min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound!")

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if theta < self.min or theta > self.max:
            return -np.inf
        else:
            return 0

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Return ``n_samples`` from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)
        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        p0 = np.exp(self.rng.uniform(low=self._log_min, high=self._log_max, size=(n_samples,)))
        return p0

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array

            The gradient of the prior at theta.
        """
        return 0


class HorseshoePrior(Prior):
    def __init__(self, scale: float, rng: np.random.RandomState):
        """Horseshoe Prior as it is used in spearmint.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        scale: float
            Scaling parameter. See below how it is influencing the distribution.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)
        self.scale = scale
        self.scale_square = scale**2

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration

        Returns
        -------
        float
        """
        # We computed it exactly as in the original spearmint code, they basically say that there's no analytical form
        # of the horseshoe prior, but that the multiplier is bounded between 2 and 4 and that they used the middle
        # See "The horseshoe estimator for sparse signals" by Carvalho, Poloson and Scott (2010), Equation 1.
        # https://www.jstor.org/stable/25734098
        # Compared to the paper by Carvalho, there's a constant multiplicator missing
        # Compared to Spearmint we first have to undo the log space transformation of the theta
        # Note: "undo log space transformation" is done in parent class
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = math.log(1 + 3.0 * (self.scale_square / theta**2))
            return math.log(a + VERY_SMALL_NUMBER)

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        # This is copied from RoBO - scale is most likely the tau parameter
        lamda = np.abs(self.rng.standard_cauchy(size=n_samples))
        p0 = np.abs(self.rng.randn() * lamda * self.scale)
        return p0

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = -(6 * self.scale_square)
            b = 3 * self.scale_square + theta**2
            b *= math.log(3 * self.scale_square * theta ** (-2) + 1)
            b = max(b, 1e-14)
            return a / b


class LognormalPrior(Prior):
    def __init__(
        self,
        sigma: float,
        rng: np.random.RandomState,
        mean: float = 0,
    ):
        """Log normal prior.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        sigma: float
            Specifies the standard deviation of the normal
            distribution.
        rng: np.random.RandomState
            Random number generator
        mean: float
            Specifies the mean of the normal distribution
        """
        super().__init__(rng)

        if mean != 0:
            raise NotImplementedError(mean)

        self.sigma = sigma
        self.sigma_square = sigma**2
        self.mean = mean
        self.sqrt_2_pi = np.sqrt(2 * np.pi)

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if theta <= self.mean:
            return -1e25
        else:
            rval = -((math.log(theta) - self.mean) ** 2) / (2 * self.sigma_square) - math.log(
                self.sqrt_2_pi * self.sigma * theta
            )
            return rval

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return self.rng.lognormal(mean=self.mean, sigma=self.sigma, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta <= 0:
            return 0
        else:
            # derivative of log(1 / (x * s^2 * sqrt(2 pi)) * exp( - 0.5 * (log(x ) / s^2))^2))
            # This is without the mean!!!
            return -(self.sigma_square + math.log(theta)) / (self.sigma_square * (theta)) * theta


class SoftTopHatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, exponent: float, rng: np.random.RandomState) -> None:
        super().__init__(rng)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.lower_bound = lower_bound
            try:
                self._log_lower_bound = np.log(lower_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w
            self.upper_bound = upper_bound
            try:
                self._log_upper_bound = np.log(upper_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w

        if exponent <= 0:
            raise ValueError("Exponent cannot be less or equal than zero (but is %f)" % exponent)
        self.exponent = exponent

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta."""
        # We need to use lnprob here instead of _lnprob to have the squared function work
        # in the logarithmic space, too.
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -((theta - self._log_lower_bound) ** self.exponent)
            elif theta > self._log_upper_bound:
                return -((self._log_upper_bound - theta) ** self.exponent)
            else:
                return 0
        else:
            raise NotImplementedError()

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return np.exp(self.rng.uniform(self._log_lower_bound, self._log_upper_bound, size=(n_samples,)))

    def gradient(self, theta: float) -> float:
        """Returns the gradient of the prior at theta."""
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -self.exponent * (theta - self._log_lower_bound)
            elif theta > self._log_upper_bound:
                return self.exponent * (self._log_upper_bound - theta)
            else:
                return 0
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return "SoftTopHatPrior(lower_bound=%f, upper_bound=%f)" % (
            self.lower_bound,
            self.upper_bound,
        )


class GammaPrior(Prior):
    def __init__(self, a: float, scale: float, loc: float, rng: np.random.RandomState):
        """Gamma prior.

        f(x) = (x-loc)**(a-1) * e**(-(x-loc)) * (1/scale)**a / gamma(a)

        Parameters
        ----------
        a: float > 0
            shape parameter
        scale: float > 0
            scale parameter (1/scale corresponds to parameter p in canonical form)
        loc: float
            mean parameter for the distribution
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)

        self.a = a
        self.loc = loc
        self.scale = scale

    def _lnprob(self, theta: float) -> float:
        """Returns the logpdf of theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) != 0:
            raise NotImplementedError()
        return sps.gamma.logpdf(theta, a=self.a, scale=self.scale, loc=self.loc)

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return self.rng.gamma(shape=self.a, scale=self.scale, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """As computed by Wolfram Alpha.

        Parameters
        ----------
        theta: float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) == 0:
            # Multiply by theta because of the chain rule...
            return ((self.a - 1) / theta - (1 / self.scale)) * theta
        else:
            raise NotImplementedError()





def get_conditional_hyperparameters(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Returns conditional hyperparameters."""
    # Taking care of conditional hyperparameters according to Levesque et al.
    X_cond = X <= -1
    if Y is not None:
        Y_cond = Y <= -1
    else:
        Y_cond = X <= -1
    active = ~((np.expand_dims(X_cond, axis=1) != Y_cond).any(axis=2))
    return active


class MagicMixin:
    # This is a mixin for a kernel to override functions of the kernel.
    # Because it overrides functions of the kernel, it needs to be placed first in the inheritance
    # hierarchy. For this reason it is not possible to subclass the
    # Mixin from the kernel class because this will prevent it from being instantiatable.
    # Therefore, mypy won't know about anything related to the superclass and I had
    # to add a few type:ignore statements when accessing a member that is declared in the
    # superclass such as self.has_conditions, self._call, super().get_params etc.

    prior = None  # type: Optional[Prior]

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Call the kernel function."""
        if active is None and self.has_conditions:  # type: ignore[attr-defined] # noqa F821
            if self.operate_on is None:
                active = get_conditional_hyperparameters(X, Y)
            else:
                if Y is None:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], None)
                else:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], Y[:, self.operate_on])

        if self.operate_on is None:
            rval = self._call(X, Y, eval_gradient, active)  # type: ignore[attr-defined] # noqa F821
        else:
            if self.len_active is None:
                raise RuntimeError("len_active is not set.")

            if Y is None:
                rval = self._call(  # type: ignore[attr-defined] # noqa F821
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=None,
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
            else:
                rval = self._call(  # type: ignore[attr-defined] # noqa F821
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=Y[:, self.operate_on].reshape([-1, self.len_active]),
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
                Y = Y[:, self.operate_on].reshape((-1, self.len_active))

        return rval

    def __add__(self, b: Union[kernels.Kernel, float]) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b: Union[kernels.Kernel, float]) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def _signature(self, func: Callable) -> Signature:
        try:
            sig_ = self._signature_cache.get(func)  # type: Optional[Signature]
        except AttributeError:
            self._signature_cache = {}  # type: Dict[Callable, Signature]
            sig_ = None
        if sig_ is None:
            sig = signature(func)
            self._signature_cache[func] = sig
            return sig
        else:
            return sig_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        try:
            args = self._args_cache
        except AttributeError:
            # ignore[misc] looks like it catches all kinds of errors, but misc is actually a category from mypy:
            # https://mypy.readthedocs.io/en/latest/error_code_list.html#miscellaneous-checks-misc
            tmp = super().get_params(deep)  # type: ignore[misc] # noqa F821
            args = list(tmp.keys())
            # Sum and Product do not clone the 'has_conditions' attribute by default. Instead of changing their
            # get_params() method, we simply add the attribute here!
            if "has_conditions" not in args:
                args.append("has_conditions")
            self._args_cache = args  # type: List[Union[str, Any]]

        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    @property
    def hyperparameters(self) -> List[kernels.Hyperparameter]:
        """Returns a list of all hyperparameter specifications."""
        try:
            return self._hyperparameters_cache
        except AttributeError:
            pass

        r = super().hyperparameters  # type: ignore[misc] # noqa F821
        self._hyperparameters_cache = r  # type: List[kernels.Hyperparameter]

        return r

    @property
    def n_dims(self) -> int:
        """Returns the number of non-fixed hyperparameters of the kernel."""
        try:
            return self._n_dims_cache
        except AttributeError:
            pass

        self._n_dims_cache = -1  # type: int
        self._n_dims_cache = super().n_dims  # type: ignore[misc] # noqa F821
        return self._n_dims_cache

    def clone_with_theta(self, theta: np.ndarray) -> kernels.Kernel:
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        """
        self.theta = theta
        return self

    def set_active_dims(self, operate_on: Optional[np.ndarray] = None) -> None:
        """Sets dimensions this kernel should work on.

        Parameters
        ----------
        operate_on : None, list or array, shape (n_dims,)
        """
        if operate_on is not None and type(operate_on) in (list, np.ndarray):
            
            if not isinstance(operate_on, np.ndarray):
                raise TypeError("argument operate_on needs to be of type np.ndarray, but is %s" % type(operate_on))
            #if operate_on.dtype != int:
            #    raise ValueError("dtype of argument operate_on needs to be int, but is %s" % operate_on.dtype)
            self.operate_on = operate_on  # type: Optional[np.ndarray]
            self.len_active = len(operate_on)  # type: Optional[int]
        else:
            self.operate_on = None
            self.len_active = None


class Sum(MagicMixin, kernels.Sum):
    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray = None,
        has_conditions: bool = False,
    ) -> None:
        super(Sum, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True, active=active)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True, active=active)
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            return self.k1(X, Y, active=active) + self.k2(X, Y, active=active)


class Product(MagicMixin, kernels.Product):
    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray = None,
        has_conditions: bool = False,
    ) -> None:
        super(Product, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True, active=active)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True, active=active)
            return K1 * K2, np.dstack((K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis]))
        else:
            return self.k1(X, Y, active=active) * self.k2(X, Y, active=active)


class ConstantKernel(MagicMixin, kernels.ConstantKernel):
    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Tuple[float, float] = (1e-5, 1e5),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[Prior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(ConstantKernel, self).__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.full(
            (X.shape[0], Y.shape[0]),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    np.full(
                        (X.shape[0], X.shape[0], 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K


class Matern(MagicMixin, kernels.Matern):
    def __init__(
        self,
        length_scale: Union[float, Tuple[float, ...], np.ndarray] = 1.0,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray] = (
            1e-5,
            1e5,
        ),
        nu: float = 1.5,
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[Prior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(Matern, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            dists = scipy.spatial.distance.pdist(X / length_scale, metric="euclidean")
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / scipy.special.gamma(self.nu))
            K *= tmp**self.nu
            K *= scipy.special.kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = scipy.spatial.distance.squareform(K)
            np.fill_diagonal(K, 1)

        if active is not None:
            K = K * active

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
            else:
                D = scipy.spatial.distance.squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            else:
                # original sklearn code would approximate gradient numerically, but this would violate our assumption
                # that the kernel hyperparameters are not changed within __call__
                raise ValueError(self.nu)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K


class RBF(MagicMixin, kernels.RBF):
    def __init__(
        self,
        length_scale: Union[float, Tuple[float, ...]] = 1.0,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (
            1e-5,
            1e5,
        ),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[Prior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(RBF, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            dists = scipy.spatial.distance.pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = scipy.spatial.distance.squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if active is not None:
            K = K * active

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * scipy.spatial.distance.squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient

        return K


class WhiteKernel(MagicMixin, kernels.WhiteKernel):
    def __init__(
        self,
        noise_level: Union[float, Tuple[float, ...]] = 1.0,
        noise_level_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (
            1e-5,
            1e5,
        ),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[Prior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(WhiteKernel, self).__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.eye(X.shape[0])

            if active is not None:
                K = K * active

            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (K, self.noise_level * np.eye(X.shape[0])[:, :, np.newaxis])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))


class HammingKernel(
    MagicMixin,
    kernels.StationaryKernelMixin,
    kernels.NormalizedKernelMixin,
    kernels.Kernel,
):
    def __init__(
        self,
        length_scale: Union[float, Tuple[float, ...], np.ndarray] = 1.0,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray] = (
            1e-5,
            1e5,
        ),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[Prior] = None,
        has_conditions: bool = False,
    ) -> None:
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    @property
    def hyperparameter_length_scale(self) -> kernels.Hyperparameter:
        """Hyperparameter of the length scale."""
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1  # type: ignore
        if anisotropic:
            return kernels.Hyperparameter("length_scale", "numeric", self.length_scale_bounds, len(length_scale))  # type: ignore  # noqa: E501
        return kernels.Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)
        Y : [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)

        K_gradient : [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.

        Note
        ----
        Code partially copied from skopt (https://github.com/scikit-optimize).
        Made small changes to only compute necessary values and use scikit-learn helper functions.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        indicator = np.expand_dims(X, axis=1) != Y
        K = (-1 / (2 * length_scale**2) * indicator).sum(axis=2)
        K = np.exp(K)

        if active is not None:
            K = K * active

        if eval_gradient:
            # dK / d theta = (dK / dl) * (dl / d theta)
            # theta = log(l) => dl / d (theta) = e^theta = l
            # dK / d theta = l * dK / dl

            # dK / dL computation
            if np.iterable(length_scale) and length_scale.shape[0] > 1:  # type: ignore
                grad = np.expand_dims(K, axis=-1) * np.array(indicator, dtype=np.float32)
            else:
                grad = np.expand_dims(K * np.sum(indicator, axis=2), axis=-1)

            grad *= 1 / length_scale**3

            return K, grad
        return K
