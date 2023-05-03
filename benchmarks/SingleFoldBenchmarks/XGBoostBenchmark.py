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
from benchmarks.MLBenchmark_Class import MLBenchmark

__version__ = '0.0.1'



class XGBoostBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 ):
        super(XGBoostBenchmark, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                'eta', lower=2**-10, upper=1., default_value=0.3, log=True
            ),  # learning rate
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=50, default_value=10, log=True
            ),
            CS.UniformFloatHyperparameter(
                'colsample_bytree', lower=0.1, upper=1., default_value=1., log=False
            ),
            CS.UniformFloatHyperparameter(
                'reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True
            ),
            CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            ),
            CS.UniformFloatHyperparameter('min_child_weight', lower=1., upper=2**7., default_value=1., log=True),
            CS.UniformFloatHyperparameter('colsample_bylevel', lower=0.01, upper=1., default_value=1.,log=False),
            CS.UniformFloatHyperparameter('reg_alpha', lower=2**-10, upper=2**10, default_value=1, log=True),
            CS.UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=500, default_value=500, log=False
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

        extra_args = dict(
            booster="gbtree",
            objective="binary:logistic",
            random_state=rng,
            eval_metric = ['auc'],
            use_label_encoder=False,
        )
        if self.n_classes > 2:
            #Very important here. We need to use softproba to get probabilities out of XGBoost
            extra_args["objective"] = 'multi:softproba' #"multi:softmax"
            extra_args.update({"num_class": self.n_classes})

        model = xgb.XGBClassifier(
            **config,
            **extra_args
        )
        return model

