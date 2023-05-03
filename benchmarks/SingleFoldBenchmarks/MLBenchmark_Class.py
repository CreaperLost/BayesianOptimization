import time
from pathlib import Path
from typing import Union, Dict, Iterable

import ConfigSpace as CS
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    precision_score, f1_score,roc_auc_score

import abc
from typing import Union, Dict
import functools

import logging
import ConfigSpace
import numpy as np

from ConfigSpace.util import deactivate_inactive_hyperparameters

from hpobench.abstract_benchmark import AbstractBenchmark
from benchmarks.data_manager import OpenMLDataManager
from benchmarks.Jad_data_manager import JadDataManager


"""acc=accuracy_score,
    bal_acc=balanced_accuracy_score,
    f1=f1_score,
    precision=precision_score,roc_auc_score""" 
    
    
    
metrics = dict(
    auc = roc_auc_score #accuracy_score
)
"""
acc=dict(),
    bal_acc=dict(),
    f1=dict(average="macro", zero_division=0),
    precision=dict(average="macro", zero_division=0),

"""



metrics_kwargs = dict(
    auc =dict(multi_class="ovr",needs_proba=True) #dict() #
)

def get_rng(rng: Union[int, np.random.RandomState, None] = None,
            self_rng: Union[int, np.random.RandomState, None] = None) -> np.random.RandomState:
    """
    Helper function to obtain RandomState from int or create a new one.

    Sometimes a default random state (self_rng) is already available, but a
    new random state is desired. In this case ``rng`` is not None and not already
    a random state (int or None) -> a new random state is created.
    If ``rng`` is already a randomState, it is just returned.
    Same if ``rng`` is None, but the default rng is given.

    Parameters
    ----------
    rng : int, np.random.RandomState, None
    self_rng : np.random.RandomState, None

    Returns
    -------
    np.random.RandomState
    """

    if rng is not None:
        return _cast_int_to_random_state(rng)
    if rng is None and self_rng is not None:
        return _cast_int_to_random_state(self_rng)
    return np.random.RandomState()


def _cast_int_to_random_state(rng: Union[int, np.random.RandomState]) -> np.random.RandomState:
    """
    Helper function to cast ``rng`` from int to np.random.RandomState if necessary.

    Parameters
    ----------
    rng : int, np.random.RandomState

    Returns
    -------
    np.random.RandomState
    """
    if isinstance(rng, np.random.RandomState):
        return rng
    if int(rng) == rng:
        # As seed is sometimes -1 (e.g. if SMAC optimizes a deterministic function) -> use abs()
        return np.random.RandomState(np.abs(rng))
    raise ValueError(f"{rng} is neither a number nor a RandomState. Initializing RandomState failed")


class MLBenchmark():
    _issue_tasks = [3917, 3945]

    def __init__(
            self,
            task_id: int,
            rng: Union[int, None] = None,
            data_path: Union[str, Path, None] = None,
            data_repo:str = 'Jad',
            use_holdout =False,
            global_seed: Union[int, None] = 1,
    ):
        
        self.global_seed = global_seed

        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        self.rng = get_rng(rng=rng)

        self.task_id = task_id
        self.scorers = dict()
        for k, v in metrics.items():
            self.scorers[k] = make_scorer(v, **metrics_kwargs[k])

        if data_path is None:
            if data_repo =='Jad':
                data_path = 'Datasets/Jad'
            else:
            #from hpobench import config_file
            #data_path = config_file.data_dir / "OpenML"
                data_path = 'Datasets/OpenML'

        self.data_path = data_path


        #Load ola ta folds.
        if data_repo == 'Jad':
            dm = JadDataManager(task_id,data_path,self.global_seed,n_folds = 5, use_holdout = use_holdout)
            dm.load()
        else:
            dm = OpenMLDataManager(task_id, data_path, self.global_seed,n_folds = 5, use_holdout = use_holdout)
            dm.load()

        # Data variables
        self.train_X = dm.train_X
        self.valid_X = dm.valid_X
        self.test_X = dm.test_X
        self.train_y = dm.train_y
        self.valid_y = dm.valid_y
        self.test_y = dm.test_y
        self.train_idx = dm.train_idx
        self.test_idx = dm.test_idx
        self.task = dm.task
        self.dataset = dm.dataset
        self.preprocessor = dm.preprocessor
        self.lower_bound_train_size = dm.lower_bound_train_size
        self.n_classes = dm.n_classes

        # Observation and fidelity spaces
        self.configuration_space = self.get_configuration_space(self.seed)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()

    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        return {
            'name': 'RF',
            'shape of train data': self.train_X[0].shape,
            'shape of test data': self.test_X.shape,
            'shape of valid data': self.valid_X[0].shape,
            'initial random seed': self.seed,
            'task_id': self.task_id
        }

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        raise NotImplementedError()

    def get_config(self, size: Union[int, None] = None):
        """Samples configuration(s) from the (hyper) parameter space
        """
        if size is None:  # return only one config
            return self.configuration_space.sample_configuration()
        return [self.configuration_space.sample_configuration() for i in range(size)]


    """def shuffle_data_idx(self, train_idx: Iterable = None, rng: Union[np.random.RandomState, None] = None) -> Iterable:
        print('This should never run')
        rng = self.rng if rng is None else rng
        train_idx = self.train_idx if train_idx is None else train_idx
        rng.shuffle(train_idx)
        return train_idx"""



    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):

        if rng is not None:
            rng = get_rng(rng, self.rng)

        

        if evaluation == "val":
            list_of_models = []
            model_fit_time = 0
            for fold in range(len(self.train_X)):
                
                # initializing model
                model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
                # preparing data -- Select the fold
                train_X = self.train_X[fold]
                train_y = self.train_y[fold]
                train_idx = self.train_idx
                # Fit the model
                start = time.time()
                
                model.fit(train_X, train_y)
                # computing statistics on training data
                model_fit_time = model_fit_time + time.time() - start
                list_of_models.append(model)

            # initializing model for the test set!
            model = self.init_model(config, fidelity, rng , n_feat = self.train_X[0].shape[1])
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)


            #Return list of models.
            model = list_of_models

            scores = dict()
            score_cost = dict()
            #Done no scoring because we just trained. :) -- Added some scores
            for k, v in self.scorers.items():
                scores[k] = 0.0
                score_cost[k] = 0.0
                _start = time.time()
                #Select model in first position.
                scores[k] = v(model[0], train_X[train_idx], train_y.iloc[train_idx])
                score_cost[k] = time.time() - _start
            train_loss = 1 - scores["auc"]
        else:
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[0].shape[1])

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            start = time.time()
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            model_fit_time = time.time() - start

            
            #This does some kind of prediction?
            # computing statistics on training data
            scores = dict()
            score_cost = dict()
            for k, v in self.scorers.items():
                _start = time.time()
                scores[k] = v(model, train_X[train_idx], train_y.iloc[train_idx])
                score_cost[k] = time.time() - _start
            train_loss = 1 - scores["auc"]

        return model, model_fit_time, train_loss, scores, score_cost


    def _check_and_cast_configuration(self,configuration: Union[Dict, ConfigSpace.Configuration],
                                      configuration_space: ConfigSpace.ConfigurationSpace) \
            -> ConfigSpace.Configuration:
        """ Helper-function to evaluate the given configuration.
            Cast it to a ConfigSpace.Configuration and evaluate if it violates its boundaries.

            Note:
                We remove inactive hyperparameters from the given configuration. Inactive hyperparameters are
                hyperparameters that are not relevant for a configuration, e.g. hyperparameter A is only relevant if
                hyperparameter B=1 and if B!=1 then A is inactive and will be removed from the configuration.
                Since the authors of the benchmark removed those parameters explicitly, they should also handle the
                cases that inactive parameters are not present in the input-configuration.
        """
        
        if isinstance(configuration, dict):
            configuration = ConfigSpace.Configuration(configuration_space, configuration,
                                                      allow_inactive_with_values=True)
        elif isinstance(configuration, ConfigSpace.Configuration):
            configuration = configuration
        else:
            raise TypeError(f'Configuration has to be from type List, np.ndarray, dict, or '
                            f'ConfigSpace.Configuration but was {type(configuration)}')

        all_hps = set(configuration_space.get_hyperparameter_names())
        active_hps = configuration_space.get_active_hyperparameters(configuration)
        inactive_hps = all_hps - active_hps

        configuration = deactivate_inactive_hyperparameters(configuration, configuration_space)
        configuration_space.check_configuration(configuration)

        return configuration

    

    def __call__(self, configuration: Dict, **kwargs) -> float:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        self._check_and_cast_configuration(configuration, self.configuration_space)
        #Get a x models trained.
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="val"
        )

        #Get the Validation Score (k-fold average)
        val_scores = dict()
        val_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            #Average validation score.
            val_scores[k] /= (len(model)-1)
            
            val_score_cost[k] = time.time() - _start
        val_loss = 1 - val_scores["auc"]


        
        #This shouldn't run in general. :)
        #Get the Test Score, once.
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            #Last model is on all the dataset and is last on the list of models. Apply it to the test-set.
            test_scores[k] = 0 #v(model[-1], self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        test_loss = 1 - test_scores["auc"]

        info = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': val_scores,
            'val_costs': val_score_cost,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': info['val_loss'],
            'cost': model_fit_time + info['val_costs']['auc'],
            'info': info
        }


    # The idea is that we run only on TEST SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """

        self._check_and_cast_configuration(configuration, self.configuration_space)


        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="test"
        )

        #If evaluation == Test then you get a single model from the train_objective :D
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        test_loss = 1 - test_scores["auc"]

        info = {
            'train_loss': train_loss,
            'val_loss': None,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': dict(),
            'val_costs': dict(),
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': float(info['test_loss']),
            'cost': float(model_fit_time + info['test_costs']['auc']),
            'info': info
        }
