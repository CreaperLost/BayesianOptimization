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
from benchmarks.Group_MultiFold_MLBenchmark_Regression import Group_MultiFold_MLBenchmark_Regression
from ConfigSpace import EqualsCondition, OrConjunction,InCondition
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import optuna
from scipy.stats import uniform,loguniform
from hyperopt import hp

__version__ = '0.0.1'


XGB_NAME = 'XGB'
RF_NAME = 'RF'
LINEAR_SVM_NAME = 'linearSVM'
RBF_SVM_NAME = 'rbfSVM'
DT_NAME = 'DT'

class Group_MultiFold_Space_Regression(Group_MultiFold_MLBenchmark_Regression):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None,
                 data_repo:str = 'Jad',
                 use_holdout =False,is_multifidelity=False
                 ):
        super(Group_MultiFold_Space_Regression, self).__init__(task_id, rng, data_path,data_repo,use_holdout,is_multifidelity=is_multifidelity)

  
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
        model = SVR(**new_config)
        return model

    def init_rf(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
        #print(config)
        model = RandomForestRegressor(min_samples_split=2,**config,  bootstrap=True,random_state=rng,n_jobs=-1)
        return model

    def init_lr(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
        # https://scikit-learn.org/stable/modules/sgd.html # performs Logistic Regression
        #print(config)
        model = SGDRegressor(**config,loss="log",learning_rate="adaptive",tol=None,random_state=rng)
        return model 
    
    def init_xgb(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None):
        extra_args = dict(
            booster="gbtree", 
            objective="reg:squarederror",
            random_state=rng,
            eval_metric = ['rmse'], #this is not used probably.
            use_label_encoder=False,
        )
         

        #Handle special case here
        new_config = config.copy() 

        new_config['max_depth'] = new_config.pop('XGB_max_depth')
        new_config['n_estimators'] = new_config.pop('XGB_n_estimators')
        #print(new_config)
        model = xgb.XGBRegressor(**new_config,**extra_args)
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

        model = DecisionTreeRegressor(min_samples_split=2,**new_config,random_state=rng)

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

    def optuna_space(self,trial,rng):

        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        # 2. Suggest values for the hyperparameters using a trial object.
        classifier_name = trial.suggest_categorical('classifier', ['LinearSVM','RbfSVM','RandomForest','DT','XGB'])
        if classifier_name == 'LinearSVM':

            linear_c = trial.suggest_float('linear_C', 2**-10, 2**10, log=True)
            model = SVR(C=linear_c)

        elif classifier_name == 'RbfSVM':

            rbf_c = trial.suggest_float('rbf_C', 2**-10, 2**10, log=True)
            rbf_gamma = trial.suggest_float('rbf_gamma', 2**-10, 2**10, log=True)
            model = SVR(C=rbf_c,gamma=rbf_gamma)

        elif classifier_name == 'DT':
            dt_max_depth = trial.suggest_int('dt_max_depth', 1, 50, log=True)
            dt_min_samples_leaf = trial.suggest_int('dt_min_samples_leaf', 1, 20, log=False)
            model = DecisionTreeRegressor(min_samples_split=2,max_depth=dt_max_depth,min_samples_leaf=dt_min_samples_leaf,random_state=rng)

        elif classifier_name == 'RandomForest':

            rf_max_depth = trial.suggest_int('rf_max_depth', 1, 50, log=True)
            rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 20, log=False)
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 100, 1000, log=False)
            rf_max_features = trial.suggest_categorical('rf_max_features', ['sqrt','log2','auto'])
            model = RandomForestRegressor(min_samples_split=2,max_depth=rf_max_depth,max_features=rf_max_features,\
                                           min_samples_leaf=rf_min_samples_leaf,  \
                                           n_estimators=rf_n_estimators, \
                                           bootstrap=True,random_state=rng,n_jobs=-1)
        
        elif classifier_name =='XGB':

            xgb_eta = trial.suggest_float('xgb_eta', 2**-10, 1, log=True)
            xgb_max_depth = trial.suggest_int('xgb_max_depth', 1, 50, log=True)
            xgb_colsample_bytree  = trial.suggest_float('xgb_colsample_bytree',0.1,1,log=False)
            xgb_reg_lambda = trial.suggest_float('xgb_reg_lambda', 2**-10, 2**10, log=True)
            xgb_subsample = trial.suggest_float('xgb_subsample', 0.1, 1, log=False)
            xgb_min_child_weight = trial.suggest_float('xgb_min_child_weight', 1, 2**7, log=True)
            xgb_colsample_bylevel = trial.suggest_float('xgb_colsample_bylevel', 0.01, 1, log=False)
            xgb_reg_alpha = trial.suggest_float('xgb_reg_alpha', 2**-10, 2**10, log=True)
            xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 500, 500, log=False)

            extra_args = dict(
                booster="gbtree",
                objective="reg:squarederror",
                random_state=rng,
                eval_metric = ['rmse'], #this is not used probably.
                use_label_encoder=False,
            )
            
            #print(new_config)
            model = xgb.XGBRegressor(eta=xgb_eta, max_depth=xgb_max_depth, 
                                      colsample_bytree = xgb_colsample_bytree,
                                      reg_lambda  = xgb_reg_lambda , reg_alpha = xgb_reg_alpha,
                                      subsample = xgb_subsample, min_child_weight = xgb_min_child_weight,
                                      colsample_bylevel = xgb_colsample_bylevel, n_estimators = xgb_n_estimators,
                                      **extra_args)
        else:
            RuntimeError

        return model 
    
    def mango_get_RF_configuration_space(self,seed) ->dict :
        param_dict_rf = {
              "min_samples_leaf": range(1, 20),
              "max_depth": range(1, 50),
              "n_estimators": range(1,1000),
              "max_features": ['sqrt','log2','auto']
             }


        return param_dict_rf
    
    def mango_get_DT_configuration_space(self,seed) -> dict:

        param_dict_dt = {
              "dt_min_samples_leaf": range(1, 20),
              "dt_max_depth": range(1, 50),
             }
        return param_dict_dt

    def mango_get_linearSVM_configuration_space(self,seed) -> dict:

        param_dict_linearSVM = {
              "linear_C": loguniform(2**-10, 2**10),
             }

        return param_dict_linearSVM

    
    def mango_get_rbfSVM_configuration_space(self,seed) -> dict:


        param_dict_rbfSVM = {
              "rbf_C": loguniform(2**-10, 2**10),
              "rbf_gamma": loguniform(2**-10, 2**10),        
             }

        return param_dict_rbfSVM
    
    def mango_get_XGB_configuration_space(self,seed: Union[int, None] = None) -> dict:
        """Parameter space to be optimized --- contains the hyperparameters
        """

        param_dict_xgboost = {
              "eta": loguniform(2**-10, 1.),
              "XGB_max_depth": range(1, 50),
              "colsample_bytree": uniform(0.1, 0.9),
              "reg_lambda": loguniform(2**-10, 2**10),
              "subsample": uniform(0.1, 0.9),
              "min_child_weight": loguniform(1., 2**7.),
              "colsample_bylevel": uniform(0.01, 0.9),
              "reg_alpha": loguniform(2**-10, 2**10),
              "XGB_n_estimators": range(50, 500),
               
             }

        return param_dict_xgboost
    
    def get_mango_config_space(self,seed=None) -> dict:
        config_dict = {}

        model_list = [XGB_NAME,LINEAR_SVM_NAME,RF_NAME,DT_NAME,RBF_SVM_NAME]


        for model_name in model_list:
            if model_name == XGB_NAME:
                xgb_space = self.mango_get_XGB_configuration_space(seed=seed)
                config_dict[XGB_NAME] = xgb_space

            elif model_name == RF_NAME:
                rf_space = self.mango_get_RF_configuration_space(seed=seed)
                config_dict[RF_NAME] = rf_space

            elif model_name == DT_NAME:
                dt_space = self.mango_get_DT_configuration_space(seed=seed)
                config_dict[DT_NAME] = dt_space

            elif model_name == LINEAR_SVM_NAME:
                linear_svm_space = self.mango_get_linearSVM_configuration_space(seed=seed)
                config_dict[LINEAR_SVM_NAME] = linear_svm_space
            
            elif model_name == RBF_SVM_NAME:
                rbf_svm_space = self.mango_get_rbfSVM_configuration_space(seed=seed)
                config_dict[RBF_SVM_NAME] = rbf_svm_space
            else:
                raise RuntimeError

        return config_dict


    def mango_init_model(self, config: Union[CS.Configuration, Dict],fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=1,model_type=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        assert model_type!=None


        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config.copy()
        
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
    


    def hyperopt_init_model(self, config: Union[CS.Configuration, Dict],
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config['algorithm'].copy()
        model_type = tmp_config.pop('type')
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



    def hyper_opt_space(self):
        
        # Define the search space for all algorithms
        search_space = {
            'algorithm': hp.choice('algorithm', [
                {
                    'type': 'XGB',
                    'eta' : hp.loguniform('eta',np.log(2**-10),np.log(1)),
                    'XGB_max_depth': hp.randint('XGB_max_depth', 1, 50),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
                    'reg_lambda' : hp.loguniform('reg_lambda',np.log(2**-10),np.log(2**10)),
                    'subsample': hp.uniform('subsample', 0.1, 1),
                    'min_child_weight': hp.loguniform('min_child_weight',np.log(1),np.log(2**7)),
                    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.01, 1),
                    'reg_alpha' : hp.loguniform('reg_alpha',np.log(2**-10),np.log(2**10)),
                    'XGB_n_estimators': hp.randint('XGB_n_estimators', 50, 500),
                },
                {
                    'type': 'RF',
                    'n_estimators': hp.randint('n_estimators', 100, 1000),
                    'max_depth': hp.randint('max_depth', 1, 50),
                    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20),
                    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
                },
                {
                    'type': 'DT',
                    'dt_max_depth': hp.randint('dt_max_depth', 1, 50),
                    'dt_min_samples_leaf': hp.randint('dt_min_samples_leaf', 1, 20),
                },
                {
                    'type': 'linearSVM',
                    'linear_C': hp.loguniform('linear_C', np.log(2**-10), np.log(2**10))
                },
                {
                    'type': 'rbfSVM',
                    'rbf_C': hp.loguniform('rbf_C', np.log(2**-10), np.log(2**10)),
                    'rbf_gamma': hp.loguniform('rbf_gamma', np.log(2**-10), np.log(2**10))
                }
            ])
        }


        return search_space
    


    def get_RF_configuration_space_multifidelity(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=20, default_value=1, log=False),
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.CategoricalHyperparameter('max_features',choices = ['sqrt','log2','auto'],default_value = 'sqrt'),
            #CS.UniformIntegerHyperparameter('n_estimators', lower=100, upper=1000, default_value=500, log=False)
        ])
        return cs
    
    def get_DT_configuration_space_multifidelity(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('dt_min_samples_leaf', lower=1, upper=20, default_value=1, log=False),
            CS.UniformIntegerHyperparameter('dt_max_depth', lower=1, upper=50, default_value=10, log=True),
        ])
        return cs

    def get_linearSVM_configuration_space_multifidelity(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("linear_C", 2**-10, 2**10, log=True, default_value=1.0),
        ])
        return cs
    
    def get_rbfSVM_configuration_space_multifidelity(self,seed) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("rbf_C", 2**-10, 2**10, log=True, default_value=1.0),
            CS.UniformFloatHyperparameter("rbf_gamma", 2**-10, 2**10, log=True, default_value=0.1),
        ])
        return cs
    
    def get_XGB_configuration_space_multifidelity(self,seed: Union[int, None] = None) -> CS.ConfigurationSpace:
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
            #CS.UniformIntegerHyperparameter('XGB_n_estimators', lower=50, upper=500, default_value=500, log=False)
        ])
        return cs
    

    def get_configuration_space_multifidelity(self,seed: Union[int, None] = None) -> CS.ConfigurationSpace:
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
                xgb_space = self.get_XGB_configuration_space_multifidelity(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=xgb_space,parent_hyperparameter={"parent": cs["model"], "value": XGB_NAME})
                config_dict[XGB_NAME] = xgb_space

            elif model_name == RF_NAME:
                rf_space = self.get_RF_configuration_space_multifidelity(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=rf_space,parent_hyperparameter={"parent": cs["model"], "value": RF_NAME})
                config_dict[RF_NAME] = rf_space

            elif model_name == DT_NAME:
                dt_space = self.get_DT_configuration_space_multifidelity(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=dt_space,parent_hyperparameter={"parent": cs["model"], "value": DT_NAME})
                config_dict[DT_NAME] = dt_space

            elif model_name == LINEAR_SVM_NAME:
                linear_svm_space = self.get_linearSVM_configuration_space_multifidelity(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=linear_svm_space,parent_hyperparameter={"parent": cs["model"], "value": LINEAR_SVM_NAME})
                config_dict[LINEAR_SVM_NAME] = linear_svm_space
            
            elif model_name == RBF_SVM_NAME:
                rbf_svm_space = self.get_rbfSVM_configuration_space_multifidelity(seed=seed)
                cs.add_configuration_space(prefix="",delimiter="",configuration_space=rbf_svm_space,parent_hyperparameter={"parent": cs["model"], "value": RBF_SVM_NAME})
                config_dict[RBF_SVM_NAME] = rbf_svm_space
            else:
                raise RuntimeError

        return cs,config_dict

   

    def init_rf_multifidelity(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None,fidelity=1):
        config['n_estimators'] = fidelity
        model = RandomForestRegressor(min_samples_split=2,**config,  bootstrap=True,random_state=rng,n_jobs=-1)
        return model

    def init_xgb_multifidelity(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None,fidelity=1):
        
        extra_args = dict(
            booster="gbtree", 
            objective="reg:squarederror",
            random_state=rng,
            eval_metric = ['rmse'], #this is not used probably.
            use_label_encoder=False,
        )
         

        #Handle special case here
        new_config = config.copy() 

        new_config['max_depth'] = new_config.pop('XGB_max_depth')
        new_config['n_estimators'] = fidelity 
        #print(new_config)
        model = xgb.XGBRegressor(**new_config,**extra_args)
        return model



    def init_model_multifidelity(self, config: Union[CS.Configuration, Dict],fidelity: int,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=1):
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()

        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config.copy()
        
        model_type = tmp_config.pop('model')
        fidelity = int(fidelity)
        if model_type == XGB_NAME:
            if fidelity ==1 :
                n_estimators = 50
            elif fidelity == 3:
                n_estimators = 200
            elif fidelity ==9 :
                n_estimators = 500
            model = self.init_xgb_multifidelity(tmp_config,rng,n_estimators)
        elif model_type == RF_NAME:
            if fidelity ==1 :
                n_estimators = 100
            elif fidelity == 3:
                n_estimators = 500
            elif fidelity ==9 :
                n_estimators = 1000
            model = self.init_rf_multifidelity(tmp_config,rng,n_estimators)
        elif model_type in [LINEAR_SVM_NAME,RBF_SVM_NAME]:
            model = self.init_svm(tmp_config,rng,model_type)
        elif model_type == DT_NAME:
            model = self.init_dt(tmp_config,rng)
        else:
            raise RuntimeError
        return model