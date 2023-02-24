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
        self.objective_time = np.array([])
        self.acquisition_time = np.array([])
        self.total_time = np.array([])

        self.inc_score = np.inf
        self.n_evals = 0
        self.verbose = True

    def run(self):
        
        configurations = self.config_space.sample_configuration(self.max_evals)
        for config in configurations:
            start_time = time()
            result_dict = self.objective_function(config.get_dictionary())
            end_time = time()-start_time
            self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))
            fX_next =np.array([result_dict['function_value']])
            
            self.fX = np.concatenate((self.fX, fX_next))
            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))
            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([0])))
            self.n_evals += 1
            if self.verbose and fX_next[0] < self.inc_score:
                self.inc_score = fX_next[0]
                print(f"{self.n_evals}) New best: {self.inc_score:.4}")

            end_time_total =  time() - start_time
            self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))

        return self.inc_score
        