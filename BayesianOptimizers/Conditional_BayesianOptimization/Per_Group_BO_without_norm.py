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
from initial_design.sobol_design import SobolDesign
from BayesianOptimizers.SMAC.Sobol_Maximizer import SobolMaximizer
"""from BayesianOptimizers.SMAC.RandomMaximizer import RandomMaximizer
from BayesianOptimizers.SMAC.MACE_Maximizer import EvolutionOpt
from BayesianOptimizers.SMAC.DE_Maximizer import DE_Maximizer
from BayesianOptimizers.SMAC.Scipy_Maximizer import Scipy_Maximizer"""
from BayesianOptimizers.SMAC.Sobol_Local_Maximizer import Sobol_Local_Maximizer

from BayesianOptimizers.SMAC.Simple_RF_surrogate import Simple_RF

"""from BayesianOptimizers.SMAC.random_forest_surrogate import RandomForest
from BayesianOptimizers.SMAC.GaussianProcess_surrogate import GaussianProcess
from BayesianOptimizers.SMAC.Hebo_Random_Forest_surrogate import HEBO_RF
from BayesianOptimizers.SMAC.Hebo_GaussianProcess_surrogate import HEBO_GP
from BayesianOptimizers.SMAC.NGBoost_surrogate import NGBoost_Surrogate
from BayesianOptimizers.SMAC.BayesianNN_surrogate import BNN_Surrogate"""


import pandas as pd



class Local_BO_Without_Norm:
    """The Random Forest Based Regression Local Bayesian Optimization.
    
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
        configuration_space:ConfigurationSpace,
        initial_design,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        random_seed = int(1e6),
        acq_funct = 'EI',
        model = 'RF',
        maximizer  = 'Sobol',group_name = '',extensive=None,STD_OUT=None
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
        assert group_name != ''
        # Save function information
        #Objective function
        self.f = f

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.config_space =  configuration_space

        # Find the type of the configuration space, either Dictionary or ConfiguratioSpace object
        # Miscellaneous
        self.isConfigspace = True if isinstance(self.config_space, ConfigurationSpace) else False
        self.hps = dict()
        if self.isConfigspace:
            for i, hp in enumerate(self.config_space.get_hyperparameters()):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i


        #If dictionary we need the following.
        if lb != None:
            self.dim = len(lb)
        else:
            self.dim = len(self.config_space.get_hyperparameters())

        self.lb = lb
        self.ub = ub

        #If configuration space just store it.
        # type: ignore[attr-defined] # noqa F821


        #Type of the initial_design, SOBOL,Random etc.
        if initial_design !=None:
            self.initial_design = initial_design
        else:
            init_design_def_kwargs = {
            "cs": self.config_space,  # type: ignore[attr-defined] # noqa F821
            "traj_logger": None,
            "rng": random_seed,
            "ta_run_limit": None,  # type: ignore[attr-defined] # noqa F821
            "configs": None,
            "n_configs_x_params": 0,
            "max_config_fracs": 0.0,
            "init_budget": n_init
            } 
            self.initial_design = SobolDesign(**init_design_def_kwargs)
        
        
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
        self.X = np.zeros((0, self.dim))
        self.X_df = pd.DataFrame()
        self.fX = np.array([])

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.checks_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 
         
        #Save the group name here in order to use on configuration objects.
        self.group_name = group_name

        # How many candidates per time. (How many Configurations to get out of Sobol Sequence)
        if extensive == None:
            self.n_cand = 900
            local_points = 100
        elif extensive == 'Hyper':
            self.n_cand = 5000
            local_points = 500
        else:
            raise RuntimeError
                


        self.model = Simple_RF(self.config_space,rng=random_seed,n_estimators=100,STD_OUT=STD_OUT)
        
        self.batch_size = 1

        if acq_funct == "EI":
            self.acquisition_function = EI(self.model)
            
            if maximizer == 'Sobol':
                self.maximize_func = SobolMaximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'Sobol_Local':
                self.maximize_func  = Sobol_Local_Maximizer(self.acquisition_function, self.config_space, self.n_cand,local_points=local_points)
            else:
                raise RuntimeError
            
        




    def vector_to_configspace(self, vector: np.array,from_normalized = True) -> ConfigurationSpace:
        '''Converts numpy array to ConfigSpace object

        Works when self.config_space is a ConfigSpace object 
        and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = impute_inactive_values(self.config_space.sample_configuration()).get_dictionary()
        #print('Vector to configuration space')
        #print(vector)
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(self.config_space.get_hyperparameters()):
            if from_normalized==True:
                if type(hyper) == OrdinalHyperparameter:
                    ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                    param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
                elif type(hyper) == CategoricalHyperparameter:
                    ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                    param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
                else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                    # rescaling continuous values
                    if hyper.log:
                        log_range = np.log(hyper.upper) - np.log(hyper.lower)
                        param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                    else:
                        param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                    if type(hyper) == UniformIntegerHyperparameter:
                        param_value = int(np.round(param_value))  # converting to discrete (int)
                    else:
                        param_value = float(param_value)
            else:
                #Really careful with LOG space...
                if type(hyper) == OrdinalHyperparameter:
                    ranges = np.arange(start=0, stop=len(hyper.sequence), step=1)
                    param_value = int(vector[i]) # hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
                elif type(hyper) == CategoricalHyperparameter:
                    ranges = np.arange(start=0, stop=len(hyper.choices), step=1)
                    param_value = hyper.choices[np.where((vector[i]  < ranges) == False)[0][-1]]
                else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                    # rescaling continuous values
                    if hyper.log:
                        log_range = np.log(hyper.upper) - np.log(hyper.lower)
                        param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                    else:
                        param_value =  float(vector[i])#hyper.lower + (hyper.upper - hyper.lower) *
                    if type(hyper) == UniformIntegerHyperparameter:
                        param_value = int(np.round(param_value))  # converting to discrete (int)
                    else:
                        param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        try:
            new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.config_space
            )
        except:
            new_config = Configuration(configuration_space=self.config_space, values = new_config,allow_inactive_with_values = True)
            #print(new_config)
        #print(new_config)
        #print('=============================================')
        return new_config


    def configspace_to_vector(self, config: ConfigurationSpace,normalize=True) -> np.array:
        '''Converts ConfigSpace object to numpy array scaled to [0,1]

        Works when self.config_space is a ConfigSpace object and the input config is a ConfigSpace object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        # the imputation replaces illegal parameter values with their default
        config = impute_inactive_values(config)
        dimensions = len(self.config_space.get_hyperparameters())
        vector = [np.nan for i in range(dimensions)]
        #print('Configuration to Vector')
        #print(config)
        for name in config:
            i = self.hps[name]

            hyper = self.config_space.get_hyperparameter(name)
            # Continuous - > infinite neighborss
            # Categorical and integers === the boundds.
            
            if normalize == True:
                if type(hyper) == OrdinalHyperparameter:
                    nlevels = len(hyper.sequence)
                    vector[i] = hyper.sequence.index(config[name]) / nlevels
                elif type(hyper) == CategoricalHyperparameter:
                    nlevels = len(hyper.choices)
                    vector[i] = hyper.choices.index(config[name]) / nlevels
                else:
                    bounds = (hyper.lower, hyper.upper)
                    param_value = config[name]
                    if hyper.log:
                        vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                    else:
                        vector[i] =  (config[name] - bounds[0]) / (bounds[1] - bounds[0])
            else:
                #print(name,config[name])
                #print(hyper)
                if type(hyper) == OrdinalHyperparameter:
                    #print('Ordinal ',hyper.sequence)
                    nlevels = len(hyper.sequence)
                    vector[i] = config[name] #hyper.sequence.index() 
                elif type(hyper) == CategoricalHyperparameter:
                    #print('Categorical ',hyper.choices)
                    nlevels = len(hyper.choices)
                    vector[i] = hyper.choices.index(config[name]) 
                else:
                    bounds = (hyper.lower, hyper.upper)
                    #print('Numerical  ',bounds)
                    param_value = config[name]
                    # CHECK 
                    if hyper.log:
                        #print('Numerical is Log')
                        vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0]) #param_value 
                        #print(f'Prev {param_value }, After log : {vector[i]}')
                    else:
                        #print('Numerical is not logged ', param_value)
                        vector[i] =  param_value

        #print(np.array(vector))
        #print("==========================")
        return np.array(vector)


    def load_initial_design_configurations(self,initial_config_size):
        if self.isConfigspace:
            # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
            population = self.initial_design._select_configurations() #self.config_space.sample_configuration(size=initial_config_size)
            if not isinstance(population, List):
                population = [population]
            # the population is maintained in a list-of-vector form where each ConfigSpace
            # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            # if no ConfigSpace representation available, uniformly sample from [0, 1]
            population = np.random.uniform(low=0.0, high=1.0, size=(initial_config_size, self.dim))
        return np.array(population)

     

    def add_group_name_to_config(self,config:Configuration):
        new_config =config.get_dictionary().copy()
        new_config['model'] = self.group_name
        return new_config

    def run_initial_configurations(self):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''

        curr_time = []
        #extra_overhead_time = time.time()
        initial_configurations = self.load_initial_design_configurations(self.n_init)
        #objective_value_per_configuration = np.array([np.inf for i in range(self.n_init)])
        #end_extra_overhead_time = time.time() - extra_overhead_time

        for i in range(self.n_init):
            time_start = time.time()
            self.run_objective(initial_configurations[i])
            #Extra timings
            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))
            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([0])))
            end_time = time.time() - time_start
            curr_time.append(end_time)
        self.total_time = np.concatenate((self.total_time,np.array(curr_time)))

    # Returns the best configuration of this specific group along with the score.
    def return_incumberment(self):
        return ( self.inc_config, self.inc_score)
    


    #Trains the surrogate model.
    def train_surrogate(self):

        start_time = time.time()

        # Warp inputs
        X = self.X  
        # Standardize values
        fX = self.fX

        #here we train...
        self.model.train(X,fX)

        #Always update the acquisition function with the new surrogate model.
        self.acquisition_function.update(self.model)

        end_time=time.time() - start_time

        #Add the extra time here.
        self.surrogate_time = np.concatenate((self.surrogate_time,np.array([end_time])))

        

    # Runs the surrogate on the points provided, computes the acquisition value per point and returns the suggested point + its value.
    def suggest_next_point(self, global_eta:float):
        
        #Currently only works with batch_size = 1 for simplicity.
        assert self.batch_size == 1

        start_time = time.time()

        #Search around my current best -- even though I compete against the global best.
        if isinstance(self.maximize_func,Sobol_Local_Maximizer):
            X_next,acquisition_value = self.maximize_func.maximize(self.configspace_to_vector,eta = global_eta,best_config = self.inc_config)
        else:
            X_next,acquisition_value = self.maximize_func.maximize(self.configspace_to_vector,eta = global_eta)

        end_time=time.time() - start_time

        #Add the acquisition time here.
        self.acquisition_time = np.concatenate((self.acquisition_time,np.array([end_time])))

        return (X_next,acquisition_value)
    


    # Checks if the current configuration is the incumberment.
    def check_if_incumberment(self,config:Configuration,fX_next:float):

        if fX_next < self.inc_score:
            self.inc_score = fX_next
            self.inc_config = config
            if self.verbose:
                print(f"{self.group_name} {self.n_evals}) New best: {self.inc_score:.4}")

    # Runs the objective function on the specified point.
    def run_objective(self,X_next:Configuration):

        assert self.batch_size == 1
        #Run objective

        start_time = time.time()
        
        ## Make sure the vector is in config_space, in order to be run fast by the model
        config = self.vector_to_configspace( X_next )

        #Run the objective function
        res = self.f(self.add_group_name_to_config(config))
        
        #Get the AUC - R2 etc.
        fX_next = res['function_value']
        #print(self.group_name,fX_next)
 
        #Increase the number of evaluations
        self.n_evals+=self.batch_size

        #Add to X and fX vectors.
        
        self.X = np.vstack((self.X, deepcopy(X_next)))
        self.fX = np.concatenate((self.fX, [fX_next]))

        #This is a better interpretable form of storing the configurations.
        new_row = pd.DataFrame(config.get_dictionary().copy(),index=[0])
        self.X_df = self.X_df.append(new_row,ignore_index=True)

        #Check if this is the best configuration.
        self.check_if_incumberment(config,fX_next)

        end_time=time.time() - start_time
        self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))

        
 
        return fX_next
     
    # Returns the best configuration of this specific group along with the score.
    def return_incumberment(self):
        return ( self.inc_config, self.inc_score)


        