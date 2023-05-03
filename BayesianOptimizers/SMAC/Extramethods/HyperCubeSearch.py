from time import time 
from ConfigSpace import ConfigurationSpace
import numpy as np
from initial_design.hyper_cube_design import LHDesign
from typing import List, Optional, Tuple


class HyperCubeSearch():


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
        
    
        init_design_def_kwargs = {
            "cs": self.config_space,  # type: ignore[attr-defined] # noqa F821
            "traj_logger": None,
            "rng": self.seed,
            "ta_run_limit": None,  # type: ignore[attr-defined] # noqa F821
            "configs": None,
            "n_configs_x_params": 0,
            "max_config_fracs": 0.0,
            "init_budget": self.max_evals
            } 
        
        
        lhs  = LHDesign(**init_design_def_kwargs)
        configurations = lhs._select_configurations() 
        if not isinstance(configurations, List):
            configurations = [configurations]
        

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
        