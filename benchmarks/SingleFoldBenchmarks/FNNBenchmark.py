"""
Changelog:
==========

0.0.1:
* First implementation of the NN Benchmarks.
"""


"""
From HPOBENCH:
https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/ml/rf_benchmark.py
Changelog:
==========
0.0.1:
* First implementation of the RF Benchmarks.
"""
import xgboost as xgb
from copy import deepcopy
from typing import Union, Tuple, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.neural_network import MLPClassifier
from benchmarks.MLBenchmark_Class import MLBenchmark

__version__ = '0.0.1'



class FNNBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 ):
        super(FNNBenchmark, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'depth', default_value=3, lower=1, upper=3, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'width', default_value=64, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=4, upper=256, default_value=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'alpha', lower=10**-8, upper=1, default_value=10**-3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'max_iter', lower=10, upper=250, default_value=250, log=False
            )
        ])
        return cs

    def init_model(self, config: Union[CS.Configuration, Dict],fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()


        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        config = deepcopy(config)
        depth = config["depth"]
        width = config["width"]
        config.pop("depth")
        config.pop("width")
        hidden_layers = [width] * depth
        model = MLPClassifier(
            **config,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",  # a fidelity being used during initialization
            random_state=rng
        )
        return model














