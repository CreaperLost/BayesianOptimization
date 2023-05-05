from copy import deepcopy
import numpy as np
import time

from ConfigSpace import ConfigurationSpace,Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace.util import impute_inactive_values,deactivate_inactive_hyperparameters

from typing import List
import typing

from acquisition_functions.ei_mine import EI
from acquisition_functions.mace import MACE

from initial_design.sobol_design import SobolDesign
from BayesianOptimizers.SMAC.Sobol_Maximizer import SobolMaximizer

from BayesianOptimizers.SMAC.Sobol_Local_Maximizer import Sobol_Local_Maximizer

from BayesianOptimizers.SMAC.Simple_RF_surrogate import Simple_RF


from BayesianOptimizers.Conditional_BayesianOptimization.per_group_smac import Per_Group_Bayesian_Optimization


import pandas as pd




class Group_Bayesian_Optimization:
    """The Random Forest Based Regression Local Bayesian Optimization.udnn-cu11, minio, kiwisolver, Jinja2, importlib-metadata, emcee, Deprecated, autograd, alive-progress, xgboost, torch, stevedore, scikit-learn, requests-toolbelt, paramz, pandas, matplotlib, george, debtcollector, dask, ConfigSpace, click, autograd-gamma, torchvision, statsmodels, pymoo, oslo.utils, oslo.config, openml, gpytorch, GPy, formulaic, 
    
    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    n_training_steps : Number of training steps for learning the GP hypers, int

    Example usage:
        RF1 = Random_Forest_1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        RF1.optimize()  # Run optimization
        X, fX = RF1.X, RF1.fX  # Evaluated points
    """


    def __init__(
        self,
        f,
        lb ,
        ub ,
        configuration_space:dict,
        initial_design,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        random_seed = int(1e6),
        acq_funct = 'EI',
        model = 'RF',
        maximizer  = 'Sobol'
    ):

        # Very basic input checks
        """assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub) 
        assert np.all(ub > lb)"""
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) 
        assert isinstance(batch_size, int)
        assert max_evals > n_init and max_evals > batch_size

        # Save function information
        #Objective function
        self.f = f

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.config_space =  configuration_space

        # Find the type of the configuration space, either Dictionary or ConfiguratioSpace object
        # Miscellaneous
        """self.isConfigspace = True if isinstance(self.config_space, ConfigurationSpace) else False
        self.hps = dict()
        if self.isConfigspace:
            for i, hp in enumerate(self.config_space.get_hyperparameters()):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i"""

        # Settings
        self.n_init = n_init #Initial configurations
        self.max_evals = max_evals #Maxmimum evaluations
        self.batch_size = batch_size #Number of points to maximize  acquisition for.
        self.verbose = verbose #hm

        # Best configuration and the score of the best configuration.
        self.inc_score = np.inf
        self.inc_config = None

        #History of actions. Complementary to the X,fx
        self.history = []


        # Save the full history
        self.X = []
        self.fX = np.array([])

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 
        
        self.batch_size = 1


        #Construct the Bayesian Optimization Objects per case.
        self.object_per_group = {}

        #Initialize the X and fX dictionaries.
        for classifier_name in configuration_space:
            classifier_specific_config_space = configuration_space[classifier_name]
            self.object_per_group[classifier_name] = Per_Group_Bayesian_Optimization(f= self.f,lb=None,ub=None,\
                                                                                    configuration_space=classifier_specific_config_space,\
                                                                                    initial_design=initial_design,n_init=n_init,max_evals=max_evals,
                                                                                    batch_size=batch_size,random_seed=random_seed,\
                                                                                      acq_funct=acq_funct,model=model,maximizer=maximizer,group_name =classifier_name )
    
    # Just call each class and run the initial configurations of each.
    def run_initial_configurations(self):
        for classifier_name in self.object_per_group:
            self.object_per_group[classifier_name].run_initial_configurations()


    # Get the initial cost.
    def compute_initial_configurations_cost(self):
        list_of_objective_time  = []
        list_of_acquisition_time  = []
        list_of_surrogate_time  = []
        list_of_checks_time = []
        list_of_total_time  = []

        for classifier_name in self.object_per_group:
            list_of_objective_time.append(self.object_per_group[classifier_name].objective_time)
            list_of_acquisition_time.append(self.object_per_group[classifier_name].acquisition_time)
            list_of_surrogate_time.append(self.object_per_group[classifier_name].surrogate_time)
            list_of_checks_time.append(self.object_per_group[classifier_name].checks_time)
            list_of_total_time.append(self.object_per_group[classifier_name].total_time)
        
        self.objective_time = np.sum(list_of_objective_time, axis=0)
        self.acquisition_time = np.sum(list_of_acquisition_time, axis=0)
        self.surrogate_time = np.sum(list_of_surrogate_time, axis=0)
        self.checks_time = np.sum(list_of_checks_time, axis=0)
        self.total_time = np.sum(list_of_total_time, axis=0)

    # Computate the best configuration.
    def compute_current_incumberment(self):
        inc_score_list = []
        for classifier_name in self.object_per_group:
            inc_score_list.append((self.object_per_group[classifier_name].inc_config , self.object_per_group[classifier_name].inc_score))
        
        # Sort the list by the first element of each tuple
        # Reverse = False means that the min is first element ( LOWEST ERROR  )
        sorted_list = sorted(inc_score_list, key=lambda x: x[1],reverse=False)
        self.inc_config = sorted_list[0][0]
        self.inc_score = sorted_list[0][1]

    # for each classifier, get the history, 
    # and keep only the best fX for each step. (To make the error curve + the config.)
    def compute_initial_configurations_curve(self):
        df = pd.DataFrame()
        #stack fX values per group
        for classifier_name in self.object_per_group:
            df[classifier_name] = self.object_per_group[classifier_name].fX
        

        
        #Compute the incumberment per step per classifer
        self.fX= np.array(df.min(axis=1))

        #Find the group with the minimum per iteration.
        min_columns = df.idxmin(axis=1)

        min_group_per_iter = list(min_columns)
        
        for i in range(len(min_group_per_iter)):
            group_name = min_group_per_iter[i]
            self.X.append(self.object_per_group[group_name].X[i])
            

    def compute_incumberment_overall(self):
        self.inc_config = self.X[np.argmin(self.fX)]
        self.inc_score = min(self.fX)
        print(f'Best score so far : {self.inc_score}')

    def run(self):

        #Run initial configurations per algorithm
        self.run_initial_configurations()
 
        # Compute costs.
        self.compute_initial_configurations_cost()

        #Compute initial_configurations_curve
        self.compute_initial_configurations_curve()

        # Compute incumberment after initial evaluations.
        self.compute_incumberment_overall()

        self.n_evals = self.n_init * len(self.object_per_group)

        #Train the surrogate models for each group(once after initial evaluations)
        for classifier_name in self.object_per_group:
            self.object_per_group[classifier_name].train_surrogate()

        while self.n_evals <= self.max_evals:
            
            #Compute acquisition per group.
            acquisition_values = []
            for classifier_name in self.object_per_group:
                X_next,acquisition_value =self.object_per_group[classifier_name].suggest_next_point(self.inc_score)
                acquisition_values.append( (classifier_name,X_next,acquisition_value) )

            
        
            #Select group with highest acquisition --> check code.
            acquisition_values_sorted = sorted(acquisition_values, key=lambda x: x[2],reverse=True)
            
            best_next_classifier = acquisition_values_sorted[0][0]
            best_next_config = acquisition_values_sorted[0][1]

            #Run objective on this group.
            fX_next = self.object_per_group[best_next_classifier].run_objective(best_next_config)

            """print(f'next best : {best_next_classifier}')
            print(f'Config: {best_next_config}')
            print(f'Acq Val: {acquisition_values_sorted[0][2]}')
            print(f'Score: {fX_next}')"""
            #Append on this the results
            self.X.append(best_next_config)
            self.fX = np.concatenate((self.fX, [fX_next]))

            #Check incumberment
            self.compute_incumberment_overall()

            #Train the surrogate model for the specific group ONLY.
            self.object_per_group[best_next_classifier].train_surrogate()

            #Increase n_evals --> current evaluations by batch-size 
            #Batch Size == 1 for now.
            self.n_evals+= self.batch_size
        for classifier_name in self.object_per_group:
            print(classifier_name, len(self.object_per_group[classifier_name].fX), min(self.object_per_group[classifier_name].fX))
        return self.inc_score


 
                


