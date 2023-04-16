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
from BayesianOptimizers.SMAC.RandomMaximizer import RandomMaximizer
from BayesianOptimizers.SMAC.MACE_Maximizer import EvolutionOpt
from BayesianOptimizers.SMAC.DE_Maximizer import DE_Maximizer
from BayesianOptimizers.SMAC.Scipy_Maximizer import Scipy_Maximizer
from BayesianOptimizers.SMAC.Sobol_Local_Maximizer import Sobol_Local_Maximizer

from BayesianOptimizers.SMAC.Simple_RF_surrogate import Simple_RF

from BayesianOptimizers.SMAC.random_forest_surrogate import RandomForest
from BayesianOptimizers.SMAC.GaussianProcess_surrogate import GaussianProcess
from BayesianOptimizers.SMAC.Hebo_Random_Forest_surrogate import HEBO_RF
from BayesianOptimizers.SMAC.Hebo_GaussianProcess_surrogate import HEBO_GP
from BayesianOptimizers.SMAC.NGBoost_surrogate import NGBoost_Surrogate
from BayesianOptimizers.SMAC.BayesianNN_surrogate import BNN_Surrogate


import pandas as pd



class Bayesian_Optimization:
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
        self.fX = np.zeros((0, 1))

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 

        # How many candidates per time. (How many Configurations to get out of Sobol Sequence)
        if 'ACQ10000' in model:
            self.n_cand = 10000
        elif 'ACQ100' in model:
            self.n_cand =100
        elif 'ACQ500' in model:
            self.n_cand =500
        else:
            self.n_cand = min(100 * self.dim, 10000)


        if 'RF' in model:
            n_est = 30
            if 'NTREE_500' in model:
                n_est = 500
            self.model = Simple_RF(self.config_space,rng=random_seed,n_estimators=n_est)
        elif 'HEBO_GP' in model:
            self.model = HEBO_GP(self.config_space,rng=random_seed)
        elif 'GP' in model:
            self.model = GaussianProcess(self.config_space,seed=random_seed)
        elif 'BNN' in model:
            self.model = BNN_Surrogate(config_space = self.config_space,rng=random_seed)
        elif 'NGBOOST' in model:
            self.model = NGBoost_Surrogate(self.config_space,rng=random_seed)


        """ elif 'HEBO_RF' in model:
            n_est = 30
            if 'NTREE_500' in model:
                n_est = 500
            self.model = HEBO_RF(self.config_space,rng=random_seed,n_estimators = n_est)"""

        if acq_funct == "EI":
            self.acquisition_function = EI(self.model)
            
            if maximizer == 'Sobol':
                self.maximize_func = SobolMaximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'Random':
                self.maximize_func = RandomMaximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'DE':
                self.maximize_func  = DE_Maximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'Scipy':
                self.maximize_func  = Scipy_Maximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'Sobol_Local':
                self.maximize_func  = Sobol_Local_Maximizer(self.acquisition_function, self.config_space, self.n_cand)
            else:
                raise RuntimeError

        elif acq_funct == "Multi5" or acq_funct == "Multi10":
            self.acquisition_function = MACE(self.model)
            self.maximize_func  = EvolutionOpt(self.config_space,self.acquisition_function,pop=100,iters=100,change_to_vector = self.configspace_to_vector)



        #Check batch_size
        if acq_funct == 'Multi5':
            self.batch_size = 5
        elif acq_funct == 'Multi10':
            self.batch_size = 10
        else:
            self.batch_size = 1
        




    def vector_to_configspace(self, vector: np.array,from_normalized = True) -> ConfigurationSpace:
        '''Converts numpy array to ConfigSpace object

        Works when self.config_space is a ConfigSpace object 
        and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = impute_inactive_values(self.config_space.sample_configuration()).get_dictionary()
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
        new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.config_space
        )
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
        for name in config:
            i = self.hps[name]

            hyper = self.config_space.get_hyperparameter(name)
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
                if type(hyper) == OrdinalHyperparameter:
                    nlevels = len(hyper.sequence)
                    vector[i] = config[name] #hyper.sequence.index() 
                elif type(hyper) == CategoricalHyperparameter:
                    nlevels = len(hyper.choices)
                    vector[i] = hyper.choices.index(config[name]) 
                else:
                    bounds = (hyper.lower, hyper.upper)
                    param_value = config[name]
                    if hyper.log:
                        vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                    else:
                        vector[i] =  param_value
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

     
    def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
        """Impute inactive hyperparameters in configurations with their default.

        Necessary to apply an EPM to the data.

        Parameters
        ----------
        configs : List[Configuration]
            List of configuration objects.

        Returns
        -------
        np.ndarray
        """
        return np.array([config.get_array() for config in configs], dtype=np.float64)

    def run_initial_configurations(self):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''



        initial_configurations = self.load_initial_design_configurations(self.n_init)
        objective_value_per_configuration = np.array([np.inf for i in range(self.n_init)])


        for i in range(self.n_init):
            start_time_total = time.time()
            
            #get the initial configuration
            
            config = self.vector_to_configspace( initial_configurations[i])
            
            #Run the objective function on it.
            start_time = time.time()
            res = self.f(config)
            end_time=time.time() - start_time
            self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))

            #Get the value and cost from objective
            #This is the validation loss averaged over all folds.
            objective_value_per_configuration[i] = res['function_value']

            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))
            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([0])))



            # If this is better than the overall best score then replace.
            if objective_value_per_configuration[i] < self.inc_score:
                self.inc_score = objective_value_per_configuration[i]
                self.inc_config = config

            end_time_total =  time.time() - start_time_total
            self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))

        #Save the new runs to both X and fX
        self.X = deepcopy(initial_configurations)
        self.fX = deepcopy(objective_value_per_configuration)
        #change the n_evals.
        self.n_evals += self.n_init



    def run(self):
        
        #Initialise and run initial configurations.
        self.run_initial_configurations()

        # Main BO loop
        while self.n_evals < self.max_evals:
            
            start_time_total = time.time()
            # Warp inputs
            X = self.X  
            # Standardize values
            fX = self.fX

    
            #Measure time as well as fitting
            start_time = time.time()

            self.model.train(X,fX)

            end_time=time.time() - start_time

            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([end_time])))

            #If we want more candidates we need to remove [0]
            self.acquisition_function.update(self.model)


            #Hm.
            start_time = time.time()
            
            
            if self.batch_size == 1:
                if isinstance(self.maximize_func,Sobol_Local_Maximizer):
                    X_next,acquistion_value = self.maximize_func.maximize(self.configspace_to_vector,eta = self.inc_score,best_config = self.inc_config)
                else:
                    X_next,acquistion_value = self.maximize_func.maximize(self.configspace_to_vector,eta = self.inc_score)
                
            else:
                X_next = self.maximize_func.maximize(initial_suggest = self.inc_config,eta = self.inc_score)
                results= pd.DataFrame(X_next).drop_duplicates()
                
                # Just add some random in here.
                if results.shape[0] < self.batch_size:
                    init_design_def_kwargs = {
                    "cs": self.config_space,  # type: ignore[attr-defined] # noqa F821
                    "traj_logger": None,
                    "rng": np.random.randint(int(1e6)) ,
                    "ta_run_limit": None,  # type: ignore[attr-defined] # noqa F821
                    "configs": None,
                    "n_configs_x_params": 0,
                    "max_config_fracs": 0.0,
                    "init_budget": self.batch_size - results.shape[0]
                    } 
                    #get some random configurations
                    r_configs = SobolDesign(**init_design_def_kwargs)._select_configurations()
                    r_cfgs = [self.configspace_to_vector(cfg) for cfg in r_configs]
                    results = results.append(r_cfgs,ignore_index=True)

                select_id = np.random.choice(results.shape[0], self.batch_size, replace = False).tolist()
                
                mu, var = self.model.predict(results)
                # smallest mean and highest variance points
                best_pred_id = np.argmin(mu)
                best_unce_id = np.argmax(var)
                #print('Mu min, max var',min(mu),max(var))
                #Add them to the first and second position.
                if best_unce_id not in select_id:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id:
                    select_id[1]= best_pred_id
                x_next_multiple = results.iloc[select_id].values.tolist()

                
            end_time=time.time() - start_time

            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([end_time])))

            #print('Acquisition time : ',self.acquisition_time[-1])

            """print('The next point selected by the AF is: ' , X_next )
            print('The acquisition value is ' , acquistion_value)"""

            #convert configuration.
            if self.batch_size ==1 :
                config = self.vector_to_configspace( X_next )
                #print('To Run Next',config )
            else:
                configs = [self.vector_to_configspace( config ) for config in x_next_multiple]
                #print('To Run Next',configs )
            
            #Run objective
            start_time = time.time()
            if self.batch_size ==1 :
                res = self.f(config)
            else:
                results = []
                for config in configs:
                    results.append(self.f(config))
            end_time=time.time() - start_time
            
            self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))
            #If we got multiple batches.
            self.n_evals+=self.batch_size



            if self.batch_size ==1:
                fX_next = [res['function_value']]

                if fX_next[0] < self.inc_score:
                    self.inc_score = fX_next[0]
                    self.inc_config = config
                    if self.verbose:
                        print(f"{self.n_evals}) New best: {self.inc_score:.4}")
                #sys.stdout.flush()

                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.concatenate((self.fX, fX_next))

            else:
                for res in results:
                    fX_next = [res['function_value']]
                    if fX_next[0] < self.inc_score:
                        self.inc_score = fX_next[0]
                        self.inc_config = config
                        if self.verbose:
                            print(f"{self.n_evals}) New best: {self.inc_score:.4}")
                #Add all feature points
                self.X = np.vstack((self.X, deepcopy(x_next_multiple)))
                #Add all result points.
                res_points = [res['function_value'] for res in results]
                self.fX = np.concatenate((self.fX, res_points))
                    

            #self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            #self.fX = np.concatenate((self.fX, fX_next))


            end_time_total =  time.time() - start_time_total
            self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))
        return self.inc_score


 
                


