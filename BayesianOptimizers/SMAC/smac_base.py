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


from BayesianOptimizers.SMAC.random_forest_surrogate import RandomForest
from BayesianOptimizers.SMAC.GaussianProcess_surrogate import GaussianProcess


from BayesianOptimizers.SMAC.kernels import (ConstantKernel,
    HammingKernel,
    Matern,
    WhiteKernel,)
from BayesianOptimizers.SMAC.utils import HorseshoePrior, LognormalPrior


class BO_RF:
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


        #Number of current evaluations!
        self.n_evals = 0 

        # Initialize parameters
        self._restart()
        self.n_cand = min(100 * self.dim, 10000)


        if model == 'RF':
            self.model = RandomForest(self.config_space,rng=random_seed)
        elif model =='GP':
            types, bounds = self.get_types(self.config_space ,instance_features=None)

            cov_amp = ConstantKernel(
                    2.0,
                    constant_value_bounds=(np.exp(-10), np.exp(2)),
                    prior=LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng),
                )

            
            cont_dims = np.where(np.array(types) == 0)[0]
            cat_dims = np.where(np.array(types) != 0)[0]
            
            if len(cont_dims) > 0:
                exp_kernel = Matern(
                        np.ones([len(cont_dims)]),
                        [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
                        nu=2.5,
                        operate_on=cont_dims,
                    )

            if len(cat_dims) > 0:
                    ham_kernel = HammingKernel(
                        np.ones([len(cat_dims)]),
                        [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
                        operate_on=cat_dims,
                    )

            assert (len(cont_dims) + len(cat_dims)) == len(self.config_space.get_hyperparameters())

            noise_kernel = WhiteKernel(
                    noise_level=1e-8,
                    noise_level_bounds=(np.exp(-25), np.exp(2)),
                    prior=HorseshoePrior(scale=0.1, rng=random_seed),
                )

            if len(cont_dims) > 0 and len(cat_dims) > 0:
                # both
                kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
            elif len(cont_dims) > 0 and len(cat_dims) == 0:
                # only cont
                kernel = cov_amp * exp_kernel + noise_kernel
            elif len(cont_dims) == 0 and len(cat_dims) > 0:
                # only cont
                kernel = cov_amp * ham_kernel + noise_kernel
            else:
                raise ValueError()

            self.model = GaussianProcess(self.config_space,types=types,bounds=bounds,kernel=kernel,seed=random_seed)
        if acq_funct == "EI":
            self.acquisition_function = EI(self.model)
        
        if maximizer == 'Sobol':
            self.maximize_func = SobolMaximizer(self.acquisition_function, self.config_space, self.n_cand)
        

    def get_types(self,
        config_space: ConfigurationSpace,
        instance_features: typing.Optional[np.ndarray] = None,
            ) -> typing.Tuple[typing.List[int], typing.List[typing.Tuple[float, float]]]:
        # Extract types vector for rf from config space and the bounds
        types = [0] * len(config_space.get_hyperparameters())
        bounds = [(np.nan, np.nan)] * len(types)

        for i, param in enumerate(config_space.get_hyperparameters()):
            parents = config_space.get_parents_of(param.name)
            if len(parents) == 0:
                can_be_inactive = False
            else:
                can_be_inactive = True

            if isinstance(param, (CategoricalHyperparameter)):
                n_cats = len(param.choices)
                if can_be_inactive:
                    n_cats = len(param.choices) + 1
                types[i] = n_cats
                bounds[i] = (int(n_cats), np.nan)

            elif isinstance(param, (OrdinalHyperparameter)):
                n_cats = len(param.sequence)
                types[i] = 0
                if can_be_inactive:
                    bounds[i] = (0, int(n_cats))
                else:
                    bounds[i] = (0, int(n_cats) - 1)

            elif isinstance(param, Constant):
                # for constants we simply set types to 0 which makes it a numerical
                # parameter
                if can_be_inactive:
                    bounds[i] = (2, np.nan)
                    types[i] = 2
                else:
                    bounds[i] = (0, np.nan)
                    types[i] = 0
                # and we leave the bounds to be 0 for now
            elif isinstance(param, UniformFloatHyperparameter):
                # Are sampled on the unit hypercube thus the bounds
                # are always 0.0, 1.0
                if can_be_inactive:
                    bounds[i] = (-1.0, 1.0)
                else:
                    bounds[i] = (0, 1.0)
            elif isinstance(param, UniformIntegerHyperparameter):
                if can_be_inactive:
                    bounds[i] = (-1.0, 1.0)
                else:
                    bounds[i] = (0, 1.0)
            elif not isinstance(param, (UniformFloatHyperparameter,
                                        UniformIntegerHyperparameter,
                                        OrdinalHyperparameter,
                                        CategoricalHyperparameter)):
                raise TypeError("Unknown hyperparameter type %s" % type(param))

        if instance_features is not None:
            types = types + [0] * instance_features.shape[1]

        return types, bounds






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

  

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.inc_score = np.inf
        self.inc_config = None
        self.history = []

    

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

    def run_initial_configurations(self, eval=True, **kwargs):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        initial_configurations = self.load_initial_design_configurations(self.n_init)
        objective_value_per_configuration = np.array([np.inf for i in range(self.n_init)])

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.n_init):
            #get the initial configuration
            
            config = self.vector_to_configspace( initial_configurations[i])
            
            #Run the objective function on it.
            res = self.f(config, **kwargs)
            #Get the value and cost from objective
            #This is the validation loss averaged over all folds.
            objective_value_per_configuration[i] = res['function_value']
            #Cost of fiting a model and evaluation of the folds.
            cost = res["cost"]
    

            info = res["info"] if "info" in res else dict()
            # If this is better than the overall best score then replace.
            if objective_value_per_configuration[i] < self.inc_score:
                self.inc_score = objective_value_per_configuration[i]
                self.inc_config = config
            #Add the new best score
            traj.append(self.inc_score)
            #Add to runtime
            runtime.append(cost)
            #Add the action to history.
            
            history.append((initial_configurations[i].tolist(), float(objective_value_per_configuration[i]), info))

        #Save the new runs to both X and fX
        self.X = deepcopy(initial_configurations)
        self.fX = deepcopy(objective_value_per_configuration)
        #change the n_evals.
        self.n_evals += self.n_init

        return traj, runtime, history


    def run(self):
        
        #Initialise and run initial configurations.
        self.run_initial_configurations()

        # Main BO loop
        while self.n_evals < self.max_evals:
            
            start_time = time.time()

            # Warp inputs
            X = self.X  # to_unit_cube(deepcopy(self._X), self.lb, self.ub)
            # Standardize values
            fX = self.fX

            """
            Output Transformation

            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
            
            """

            #Train surrogate model.
            #Train the random forest
            
            self.model.train(X,fX)

            #If we want more candidates we need to remove [0]

            self.acquisition_function.update(self.model)

            X_next = self.maximize_func.maximize(self.configspace_to_vector,eta = self.inc_score)

            fX_next = [self.f(self.vector_to_configspace(X_next))['function_value']]

            self.n_evals+=1

            if self.verbose and fX_next[0] < self.inc_score:
                self.inc_score = fX_next[0]
                print(f"{self.n_evals}) New best: {self.inc_score:.4}")
                #sys.stdout.flush()

            self.X = np.vstack((self.X, deepcopy(X_next)))
            
            #self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self.fX = np.concatenate((self.fX, fX_next))



        return self.inc_score


 
                

