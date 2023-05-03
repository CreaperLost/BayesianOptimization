"""
From HPOBENCH:
https://github.com/automl/HPOBench/blob/master/hpobench/benchmarks/ml/rf_benchmark.py
Changelog:
==========
0.0.1:
* First implementation of the RF Benchmarks.
"""
from typing import Union, Dict 
import ConfigSpace as CS
import numpy as np
from benchmarks.Group_MultiFold_MLBenchmark import Group_MultiFold_MLBenchmark
from ConfigSpace import EqualsCondition, OrConjunction,InCondition
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

__version__ = '0.0.1'


XGB_NAME = 'XGB'
RF_NAME = 'RF'
LINEAR_SVM_NAME = 'linearSVM'
RBF_SVM_NAME = 'rbfSVM'
DT_NAME = 'DT'

class Group_MultiFold_Space(Group_MultiFold_MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False
                 ):
        super(Group_MultiFold_Space, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

  
    def get_RF_configuration_space(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=20, default_value=1, log=False),
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','auto'],default_value = 'sqrt'),
            CS.UniformIntegerHyperparameter('n_estimators', lower=100, upper=1000, default_value=500, log=False)
        ])
        return cs
    
    def get_DT_configuration_space(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('dt_min_samples_leaf', lower=1, upper=20, default_value=1, log=False),
            CS.UniformIntegerHyperparameter('dt_max_depth', lower=1, upper=50, default_value=10, log=True),
        ])
        return cs

    def get_linearSVM_configuration_space(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("linear_C", 2**-10, 2**10, log=True, default_value=1.0),
        ])
        return cs
    
    def get_rbfSVM_configuration_space(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("rbf_C", 2**-10, 2**10, log=True, default_value=1.0),
            CS.UniformFloatHyperparameter("rbf_gamma", 2**-10, 2**10, log=True, default_value=0.1),
        ])
        return cs
    
    def get_XGB_configuration_space(self,seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('eta', lower=2**-10, upper=1., default_value=0.3, log=True),  # learning rate
            CS.UniformIntegerHyperparameter('XGB_max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.UniformFloatHyperparameter('colsample_bytree', lower=0.1, upper=1., default_value=1., log=False),
            CS.UniformFloatHyperparameter('reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True),
            CS.UniformFloatHyperparameter('subsample', lower=0.1, upper=1, default_value=1, log=False),
            CS.UniformFloatHyperparameter('min_child_weight', lower=1., upper=2**7., default_value=1., log=True),
            CS.UniformFloatHyperparameter('colsample_bylevel', lower=0.01, upper=1., default_value=1.,log=False),
            CS.UniformFloatHyperparameter('reg_alpha', lower=2**-10, upper=2**10, default_value=1, log=True),
            CS.UniformIntegerHyperparameter(
                'XGB_n_estimators', lower=50, upper=500, default_value=500, log=False
            )
        ])
        return cs
    
    
    def get_configuration_space(self,seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        config_dict = {}

        model_list = [XGB_NAME,LINEAR_SVM_NAME,RF_NAME,DT_NAME,RBF_SVM_NAME]
        cs.add_hyperparameters([CS.CategoricalHyperparameter('model', choices = model_list,default_value='XGB')])
        
        # We set the prefix and delimiter to be empty string "" so that we don't have to do
        # any extra parsing once sampling

        for model_name in model_list:
            if model_name == XGB_NAME:
                xgb_space = self.get_XGB_configuration_space(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=xgb_space,parent_hyperparameter={"parent": cs["model"], "value": XGB_NAME})
                config_dict[XGB_NAME] = xgb_space

            elif model_name == RF_NAME:
                rf_space = self.get_RF_configuration_space(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=rf_space,parent_hyperparameter={"parent": cs["model"], "value": RF_NAME})
                config_dict[RF_NAME] = rf_space

            elif model_name == DT_NAME:
                dt_space = self.get_DT_configuration_space(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=dt_space,parent_hyperparameter={"parent": cs["model"], "value": DT_NAME})
                config_dict[DT_NAME] = dt_space

            elif model_name == LINEAR_SVM_NAME:
                linear_svm_space = self.get_linearSVM_configuration_space(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=linear_svm_space,parent_hyperparameter={"parent": cs["model"], "value": LINEAR_SVM_NAME})
                config_dict[LINEAR_SVM_NAME] = linear_svm_space
            
            elif model_name == RBF_SVM_NAME:
                rbf_svm_space = self.get_rbfSVM_configuration_space(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=rbf_svm_space,parent_hyperparameter={"parent": cs["model"], "value": RBF_SVM_NAME})
                config_dict[RBF_SVM_NAME] = rbf_svm_space
            else:
                raise RuntimeError

        return cs,config_dict


    def init_svm(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None,model_type = LINEAR_SVM_NAME):

        new_config = config.copy()
        if model_type == LINEAR_SVM_NAME:
            new_config['C'] = new_config.pop('linear_C')
        elif model_type == RBF_SVM_NAME:
            new_config['C'] = new_config.pop('rbf_C')
            new_config['gamma'] = new_config.pop('rbf_gamma')
        else:
            raise RuntimeError
            """new_config['C'] = new_config.pop('poly_C')
            new_config['gamma'] = new_config.pop('poly_gamma')
            new_config['degree'] = new_config.pop('poly_degree')
            new_config['coef0'] = new_config.pop('poly_coef0')"""
        # initializing model           
        #print(new_config)
        model = SVC(**new_config,random_state=rng,probability=True)
        return model

    def init_rf(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
        #print(config)
        model = RandomForestClassifier(min_samples_split=2,**config,  bootstrap=True,random_state=rng,n_jobs=-1)
        return model

    def init_lr(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
        # https://scikit-learn.org/stable/modules/sgd.html # performs Logistic Regression
        #print(config)
        model = SGDClassifier(**config,loss="log",learning_rate="adaptive",tol=None,random_state=rng)
        return model 
    
    def init_xgb(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
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

        #Handle special case here
        new_config = config.copy() 

        new_config['max_depth'] = new_config.pop('XGB_max_depth')
        new_config['n_estimators'] = new_config.pop('XGB_n_estimators')
        #print(new_config)
        model = xgb.XGBClassifier(**new_config,**extra_args)
        return model


    def init_dt(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None , n_feat = 1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        new_config = config.copy()
       
        new_config['max_depth'] = new_config.pop('dt_max_depth')
        new_config['min_samples_leaf'] = new_config.pop('dt_min_samples_leaf')

        model = DecisionTreeClassifier(min_samples_split=2,**config,random_state=rng)

        return model


    def init_model(self, config: Union[CS.Configuration, Dict],fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()


        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config.copy()
        
        model_type = tmp_config.pop('model')

        if model_type == XGB_NAME:
            model = self.init_xgb(tmp_config,rng)
        elif model_type == RF_NAME:
            model = self.init_rf(tmp_config,rng)
        elif model_type in [LINEAR_SVM_NAME,RBF_SVM_NAME]:
            model = self.init_svm(tmp_config,rng,model_type)
        elif model_type == DT_NAME:
            model = self.init_dt(tmp_config,rng)
        else:
            raise RuntimeError
        return model

