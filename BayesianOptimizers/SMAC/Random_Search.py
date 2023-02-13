from time import time 
from ConfigSpace import ConfigurationSpace
import numpy as np



class Random_Search():


    def __init__(self,
        f,
        configuration_space:ConfigurationSpace,
        n_init,
        max_evals,
        random_seed = int(1e6),verbose=True):

        self.objective_function = f 
        self.config_space = configuration_space

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.n_init = n_init #Initial configurations
        self.max_evals = max_evals #Maxmimum evaluations

        # Save the full history
        self.fX = np.array([])

        self.surrogate_time = np.array([])

        self.inc_score = np.inf
        self.n_evals = 0
        self.verbose = True

    def run(self):
        
        configurations = self.config_space.sample_configuration(self.max_evals)
        for config in configurations:
            result_dict = self.objective_function(config.get_dictionary())
            fX_next =np.array([result_dict['function_value']])
            
            self.fX = np.concatenate((self.fX, fX_next))
            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))

            self.n_evals += 1
            if self.verbose and fX_next[0] < self.inc_score:
                self.inc_score = fX_next[0]
                print(f"{self.n_evals}) New best: {self.inc_score:.4}")

        return self.inc_score
        