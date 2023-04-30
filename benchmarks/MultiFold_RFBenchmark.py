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
from benchmarks.MultiFold_MLBenchmark import MultiFold_MLBenchmark
from sklearn.ensemble import RandomForestClassifier

__version__ = '0.0.1'



class MultiFold_RFBenchmark(MultiFold_MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 ):
        super(MultiFold_RFBenchmark, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=20, default_value=1, log=False),
            CS.UniformIntegerHyperparameter('min_samples_split',lower=2, upper=128, default_value=32, log=True),
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','auto'],default_value = 'sqrt'),
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

        model = RandomForestClassifier(n_estimators=250,**config,  bootstrap=True,random_state=rng,n_jobs=-1)
        return model

