"""
From HPOBENCH:
https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/ml/rf_benchmark.py
Changelog:
==========
0.0.1:
* First implementation of the RF Benchmarks.
"""
import xgboost as xgb
from typing import Union, Dict
import ConfigSpace as CS
import numpy as np
from benchmarks.MultiFoldBenchmarks.MultiFold_MLBenchmark import MultiFold_MLBenchmark
from sklearn.svm import SVC


__version__ = '0.0.1'



class MultiFold_LinearSVMBenchmark(MultiFold_MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 ):
        super(MultiFold_LinearSVMBenchmark, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("C", 2**-10, 2**10, log=True, default_value=1.0)
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

        #print(new_config)
        model = SVC(**config,random_state=rng,probability=True)
        return model

