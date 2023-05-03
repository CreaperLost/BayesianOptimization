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
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 
                 ):
        super(RandomForestBenchmark, self).__init__(task_id, rng, data_path,data_repo,use_holdout) 

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('min_samples_leaf',lower = 1,upper = 10 ,q=3,default_value = 1),
            CS.UniformIntegerHyperparameter('min_samples_split',lower = 2,upper = 16,q=2,default_value = 2),
            CS.UniformIntegerHyperparameter('max_depth',lower = 3,upper = 30,q=3,default_value = 30),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','None'],default_value = 'sqrt'),
        ])
        return cs

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
            n_estimators=100,
            #n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            n_jobs=-1,
            random_state=rng
        )
        return model

