
import numpy as np


from BayesianOptimizers.SMAC.base_surrogate_model import BaseModel,get_types
from pyrfr import regression
from typing import List, Optional, Tuple
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace.util import impute_inactive_values,deactivate_inactive_hyperparameters
import typing

try:
    import pyrfr.regression as reg
except:
    raise ValueError("If you want to use Random Forests you have to install the following dependencies:\n"
                     "Pyrfr (pip install pyrfr)")
 

class RandomForest(BaseModel):

    def __init__(self, config_space, num_trees=30,
                 do_bootstrapping=True, 
                 n_points_per_tree=0,
                 compute_oob_error=False,
                 return_total_variance=True,
                 rng=None):
        """
        Interface for the random_forest_run library to model the
        objective function with a random forest.

        Parameters
        ----------
        num_trees: int
            The number of trees in the random forest.
        do_bootstrapping: bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree: int
            Number of data point per tree. If set to 0 then we will use all data points in each tree
        compute_oob_error: bool
            Turns on / off calculation of out-of-bag error. Default: False
        return_total_variance: bool
            Return law of total variance (mean of variances + variance of means, if True)
            or explained variance (variance of means, if False). Default: True
        rng: np.random.RandomState
            Random number generator 
        """
        self.config_space = config_space


        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.reg_rng = reg.default_random_engine(self.rng)

        self.n_points_per_tree = n_points_per_tree

        
        self.rf = reg.binary_rss_forest()
        self.rf.options.num_trees = num_trees
        self.rf.options.do_bootstrapping = do_bootstrapping
        self.rf.options.num_data_points_per_tree = n_points_per_tree
        self.rf.options.compute_oob_error = compute_oob_error
        self.rf.options.compute_law_of_total_variance = return_total_variance
        self.log_y = False

        self.normalize_y = True
        super(RandomForest, self).__init__()


    

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> regression.default_data_container:
        """Fills a pyrfr default data container, s.t. the forest knows categoricals and bounds for
        continous data.
        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values
        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        self.types, self.bounds = get_types(self.config_space, None)
        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data



    def train(self, X, y, **kwargs):
        
        """
        Trains the random forest on X and y.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        X = self._impute_inactive(X)
        if self.normalize_y:
            y = self._normalize_y(y)
        y = y.flatten()
        if self.n_points_per_tree == 0:
            self.rf.options.num_data_points_per_tree = X.shape[0]
        


        data = self._init_data_container(X ,y)
        self.rf.fit(data = data, rng=regression.default_random_engine(self.rng) )
        

    def predict(self, X: np.ndarray, cov_return_type="diagonal_cov") -> Tuple[np.ndarray, np.ndarray]:
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
        """if X.shape[1] != len(self.types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self.types), X.shape[1]))"""
        if cov_return_type != "diagonal_cov":
            raise ValueError("'cov_return_type' can only take 'diagonal_cov' for this model")

        if self.log_y:
            all_preds = []
            third_dimension = 0

            # Gather data in a list of 2d arrays and get statistics about the required size of the 3d array
            for row_X in X:
                preds_per_tree = self.rf.all_leaf_values(row_X)
                all_preds.append(preds_per_tree)
                max_num_leaf_data = max(map(len, preds_per_tree))
                third_dimension = max(max_num_leaf_data, third_dimension)

            # Transform list of 2d arrays into a 3d array
            preds_as_array = np.zeros((X.shape[0], self.rf_opts.num_trees, third_dimension)) * np.NaN
            for i, preds_per_tree in enumerate(all_preds):
                for j, pred in enumerate(preds_per_tree):
                    preds_as_array[i, j, : len(pred)] = pred

            # Do all necessary computation with vectorized functions
            preds_as_array = np.log(np.nanmean(np.exp(preds_as_array), axis=2) + 0.00001)

            # Compute the mean and the variance across the different trees
            means = preds_as_array.mean(axis=1)
            vars_ = preds_as_array.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                mean_, var = self.rf.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        if self.normalize_y:
            means, vars_ = self._untransform_y(means, vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))


    def predict_each_tree(self, X_test, **args):
        pass

    def sample_functions(self, X_test, n_funcs=1):
        pass

    def __getstate__(self):
        sdict = self.__dict__.copy()
        del sdict['reg_rng']  # delete not-pickleable objects
        return sdict

    def __setstate__(self, sdict):
         self.__dict__.update(sdict)
         self.reg_rng = reg.default_random_engine(sdict['rng'].randint(1000))
