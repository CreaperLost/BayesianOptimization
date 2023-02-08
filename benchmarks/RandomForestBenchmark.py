"""
From HPOBENCH:
https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/ml/rf_benchmark.py
Changelog:
==========
0.0.1:
* First implementation of the RF Benchmarks.
"""

from copy import deepcopy
from typing import Union, Tuple, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.ensemble import RandomForestClassifier

from benchmarks.MLBenchmark_Class import MLBenchmark

__version__ = '0.0.1'


class RandomForestBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(RandomForestBenchmark, self).__init__(task_id, rng, valid_size, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.OrdinalHyperparameter('n_estimators',sequence = [50,100,250],default_value = 100),
            CS.OrdinalHyperparameter('min_samples_leaf',sequence = [1,3,5,10],default_value = 1),
            CS.OrdinalHyperparameter('min_samples_split',sequence = [2,4,8,16],default_value = 2),
            CS.OrdinalHyperparameter('max_depth',sequence = [3,10,25,5000],default_value = 5000),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','None'],default_value = 'sqrt'),
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            RandomForestBenchmark._get_fidelity_choices(n_estimators_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(n_estimators_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        assert n_estimators_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed= CS.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=250, default_value=100, log=False
            ),
            variable=CS.UniformIntegerHyperparameter(
                'n_estimators', lower=16, upper=512, default_value=512, log=False
            )
        )

        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        n_estimators = fidelity1[n_estimators_choice]
        subsample = fidelity2[subsample_choice]
        return n_estimators, subsample

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None , n_feat = 1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary() 
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        config = deepcopy(config)
        #n_features = n_feat #self.train_X.shape[1]
        #config["max_features"] = int(np.rint(np.power(n_features, config["max_features"])))
        if config['max_depth'] == 'None':
            config['max_depth'] = None 
        if config['max_features'] == 'None':
            config['max_features'] = None 
        model = RandomForestClassifier(
            **config,
            #n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            n_jobs=-1,
            random_state=rng
        )
        return model


class RandomForestBenchmarkBB(RandomForestBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        return fidelity_space


"""class RandomForestBenchmarkMF(RandomForestBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - ntrees
            RandomForestBenchmark._get_fidelity_choices(n_estimators_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space
"""


class RandomForestBenchmark_2(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(RandomForestBenchmark_2, self).__init__(task_id, rng, valid_size, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators',lower = 50,upper = 250,q=5,default_value = 100),
            CS.UniformIntegerHyperparameter('min_samples_leaf',lower = 1,upper = 10 ,q=3,default_value = 1),
            CS.UniformIntegerHyperparameter('min_samples_split',lower = 2,upper = 16,q=2,default_value = 2),
            CS.UniformIntegerHyperparameter('max_depth',lower = 3,upper = 30,q=3,default_value = 30),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','None'],default_value = 'sqrt'),
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            RandomForestBenchmark._get_fidelity_choices(n_estimators_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(n_estimators_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        assert n_estimators_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed= CS.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=250, default_value=100, log=False
            ),
            variable=CS.UniformIntegerHyperparameter(
                'n_estimators', lower=16, upper=512, default_value=512, log=False
            )
        )

        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        n_estimators = fidelity1[n_estimators_choice]
        subsample = fidelity2[subsample_choice]
        return n_estimators, subsample

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None , n_feat = 1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        config = deepcopy(config)
        #n_features = n_feat #self.train_X.shape[1]
        #config["max_features"] = int(np.rint(np.power(n_features, config["max_features"])))
        if config['max_depth'] == 'None':
            config['max_depth'] = None 
        if config['max_features'] == 'None':
            config['max_features'] = None 
        model = RandomForestClassifier(
            **config,
            #n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            n_jobs=-1,
            random_state=rng
        )
        return model

class RandomForestBenchmarkBB_2(RandomForestBenchmark_2):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        """fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            RandomForestBenchmark._get_fidelity_choices(n_estimators_choice='fixed', subsample_choice='fixed')
        )"""
        return fidelity_space