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
"""from BayesianOptimizers.SMAC.ExtraMethods.RandomMaximizer import RandomMaximizer
from BayesianOptimizers.SMAC.MACE_Maximizer import EvolutionOpt
from BayesianOptimizers.SMAC.DE_Maximizer import DE_Maximizer
from BayesianOptimizers.SMAC.Scipy_Maximizer import Scipy_Maximizer"""
from BayesianOptimizers.SMAC.Sobol_Local_Maximizer import Sobol_Local_Maximizer
from BayesianOptimizers.SMAC.Simple_RF_surrogate import Simple_RF

"""
from BayesianOptimizers.SMAC.random_forest_surrogate import RandomForest
from BayesianOptimizers.SMAC.GaussianProcess_surrogate import GaussianProcess
from BayesianOptimizers.SMAC.Hebo_Random_Forest_surrogate import HEBO_RF
from BayesianOptimizers.SMAC.Hebo_GaussianProcess_surrogate import HEBO_GP
from BayesianOptimizers.SMAC.NGBoost_surrogate import NGBoost_Surrogate
from BayesianOptimizers.SMAC.BayesianNN_surrogate import BNN_Surrogate
"""

import pandas as pd



class Bayesian_Optimization_MultiFold:
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
        maximizer  = 'Sobol',
        n_folds = 10,
        per_fold = 1,
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


        self.n_folds = n_folds
        #Run BO per folds.
        self.per_fold = per_fold

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
        #max evaluations per fold. if we got 10 folds, 500 total then we do 50 per fold. but if we go every 2 folds, we can do 100.
        self.max_evals_per_fold = (self.per_fold * self.max_evals ) / self.n_folds
        #Ensure in a loop that the batch_size won't exceed the allowed iterations per fold.
        assert self.max_evals_per_fold >= batch_size
        self.batch_size = batch_size #Number of points to maximize  acquisition for.
        self.verbose = verbose #hm

        # Best configuration and the score of the best configuration.
        self.inc_score = np.inf
        self.inc_config = None

        
        #Keep score of each fold in here...
        self.y = [list() for _ in range(self.n_folds)]


        # Save the full history
        self.X = np.zeros((0, self.dim))
        #Keep the average in self.fX
        self.fX = np.zeros((0, 1))

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 

        # How many candidates per time. (How many Configurations to get out of Sobol Sequence)
   
        self.n_cand = min(100 * self.dim, 10000)


        self.model = Simple_RF(self.config_space,rng=random_seed,n_estimators=100)
        
        
     
        

        if acq_funct == "EI":
            self.acquisition_function = EI(self.model)
            
            if maximizer == 'Sobol':
                self.maximize_func = SobolMaximizer(self.acquisition_function, self.config_space, self.n_cand)
            elif maximizer == 'Sobol_Local':
                self.maximize_func  = Sobol_Local_Maximizer(self.acquisition_function, self.config_space, self.n_cand)
            else:
                raise RuntimeError


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
        try:
            new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.config_space
            )
        except:
            new_config = Configuration(configuration_space=self.config_space, values = new_config,allow_inactive_with_values = True)
            print(new_config)
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


    # Run some random initial configurations on fold 
    def run_initial_configurations(self,fold=None):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        assert fold!=None

        initial_configurations = self.load_initial_design_configurations(self.n_init)
        objective_value_per_configuration = np.array([np.inf for i in range(self.n_init)])


        for i in range(self.n_init):
            start_time_total = time.time()
            
            #get the initial configuration
            
            config = self.vector_to_configspace( initial_configurations[i])
            
            #Run the objective function on it.
            start_time = time.time()
            
            objective_value_per_configuration[i] = self.f(config,fold=fold)['function_value']

            end_time=time.time() - start_time

            self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))

            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))
            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([0])))

            # If this is better than the overall best score then replace.
            if objective_value_per_configuration[i] < self.inc_score:
                self.inc_score = objective_value_per_configuration[i]
                self.inc_config = config

            end_time_total =  time.time() - start_time_total
            self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))


        #Sanity check.
        self.y[fold] = (self.y[fold] + objective_value_per_configuration.flatten().tolist())
        
        #Save the new runs to both X and fX
        self.X = deepcopy(initial_configurations)
        self.fX = deepcopy(objective_value_per_configuration)
        #change the n_evals.
        self.n_evals += self.n_init


    def run_on_current_fold(self,fold):
        #Run the previous on the new fold. and add the results to the list
        self.y[fold] = [self.f(self.vector_to_configspace( config ),fold=fold)['function_value'] for config in self.X]

    # This will be gready, as we should only care about the current fold avg. Not the previous
    def compute_avg_performance(self,iter_fold):
        # Store the current predictions in np.array
        # Get the mean of each ROW (Config)
        # Store in fX :)
        
        #print(np.array([self.y[i] for i in range(iter_fold)]).mean(axis=0).shape)
        self.fX = np.array([self.y[i] for i in range(iter_fold)]).mean(axis=0)


    def compute_current_inc_after_avg(self):
        self.inc_score = np.min(self.fX)
        #Here we store a config space object
        self.inc_config = self.vector_to_configspace(self.X[np.argmin(self.fX)])
        if self.verbose:
            print(f"{self.n_evals}) New best: {self.inc_score:.4}")


    # Get the score per fold, and return the avg.
    def run_objective_on_previous_folds(self,config,iter_fold):
        #again this is the iterator fold, so its up-to. Fold 0 == Iterator Fold 1.
        per_fold_auc = [self.f(config,fold=f)['function_value'] for f in range(iter_fold)]
        # each list increase by 1 config for each fold.
        for f in range(iter_fold):
            self.y[f] = self.y[f] + [per_fold_auc[f]]
        return np.mean(per_fold_auc)
    
    # Save the performance per fold. Per self.max_evals_per_fold...
    def save_performance_history_between_folds(self):
        pass

    def run(self):
        

        for fold in range(0,self.n_folds):
            print('Currently Optimizing Fold  : ' , fold)
            assert self.n_evals <= self.max_evals
            #print(f"Current fold : {fold}, Total Folds {self.n_folds}")
            #print(f"N_Evals : {self.n_evals}, Maximum Evals {self.max_evals}")
            #Run the initial configurations (Sobol) for first fold.
            #Only runs ONCE.

            iterator_fold = fold+1
            if fold==0:
                self.run_initial_configurations(fold=fold)
            #print('Run Just Initials.')
            #print('Len X, Len FX',len(self.X),len(self.fX))
            #print(f"Iterator Fold : {iterator_fold}")
            # Prota run ta configurations eos tora sto 2o fold ktlp kai average out.
            
            re_run_cost_start =  time.time()
            #From the 2nd fold onwards do this for the previous configurations
            if fold != 0 :
                
                #Run the previous optimized configurations on current fold.
                self.run_on_current_fold(fold)

                #After running the previous on the new fold.
                #We have to get some average scores.
                self.compute_avg_performance(iterator_fold)

                #And compute the incumberment on the folds using an averaging., before running BO again
                self.compute_current_inc_after_avg()
            re_run_cost_end = time.time() - re_run_cost_start
            
            #Initial Overhead for next fold
            

            """After training everything on the specified folds run BO.
                #Recheck for the per x case.

                #Measure time as well as fitting
                #max_evals_per_fold per fold iteration, forxample 50 per 1 fold, 100 per 2 folds etc."""
            curr_eval = 0
            # Only for the first fold, we have some initials.
            if fold == 0:
                curr_eval = self.n_init
                re_run_cost_end = 0


            while curr_eval< self.max_evals_per_fold:
                start_time_total = time.time()
                # Just set tmp input
                X = self.X  
                # Get the current input
                fX = self.fX

                #print('Len X, Len FX',len(self.X),len(self.fX))
                # Train the surrogate
                start_time = time.time()

                self.model.train(X,fX)

                end_time=time.time() - start_time
                #print('Surrogate time',end_time)
                self.surrogate_time = np.concatenate((self.surrogate_time,np.array([end_time])))

                #If we want more candidates we need to remove [0]
                self.acquisition_function.update(self.model)

                #Start the Acquisition maximization procedure.
                start_time = time.time()
                    
                if isinstance(self.maximize_func,Sobol_Local_Maximizer):
                    X_next,acquistion_value = self.maximize_func.maximize(self.configspace_to_vector,eta = self.inc_score,best_config = self.inc_config)
                else:
                    X_next,acquistion_value = self.maximize_func.maximize(self.configspace_to_vector,eta = self.inc_score)
                        
                end_time=time.time() - start_time
                self.acquisition_time = np.concatenate((self.acquisition_time,np.array([end_time])))

                #print('Acquisition time',end_time)

                #Run objective -- SANITY CHECK
                start_time = time.time()
                
                #Turn the vector [0,1] to a configspace object.
                config = self.vector_to_configspace( X_next )
                fX_next = [self.run_objective_on_previous_folds(config,iterator_fold)]
                
                end_time=time.time() - start_time

                #print('Objective_Time',end_time)
                self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))


                #Check if better than current incumberment.

                if fX_next[0] < self.inc_score:
                    self.inc_score = fX_next[0]
                    self.inc_config = config
                    if self.verbose:
                        print(f"{self.n_evals}) New best: {self.inc_score:.4}")

                #Keep the curr_eval infold iterator updated
                #keep n_evals global iterator updated
                self.n_evals += self.batch_size
                curr_eval += self.batch_size

                # This is fine
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.concatenate((self.fX, fX_next))

                end_time_total =  time.time() - start_time_total

                #If we got initial cost.
                if re_run_cost_end != 0:    
                    #Add it to the total time.
                    end_time_total+= re_run_cost_end
                    #make it zero till we move to next fold.
                    re_run_cost_end = 0
                    
                #print('Total_Time',end_time_total)
                self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))
                
        return self.inc_score
    