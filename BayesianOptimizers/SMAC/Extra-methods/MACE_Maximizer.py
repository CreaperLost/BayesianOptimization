# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy  as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from pymoo.factory import get_problem, get_mutation, get_crossover, get_algorithm
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.config import Config
Config.show_compile_hint = False
from initial_design.sobol_design import SobolDesign
from typing import List, Optional, Tuple
from BayesianOptimizers.SMAC.base_surrogate_model import get_types

class BOProblem(Problem):
    def __init__(self,
            lb    : np.ndarray,
            ub    : np.ndarray,
            acq   ,
            space , 
            eta,
            ):
        super().__init__(len(lb), xl = lb, xu = ub, n_obj = acq.num_obj, n_constr = acq.num_constr)
        self.acq   = acq
        self.space = space
        self.eta = eta


    #Here we evaluate the acquisition functions for each input point... :)
    def _evaluate(self, X : np.ndarray, out : dict, *args, **kwargs):
        num_x = X.shape[0]
        
        with torch.no_grad():
            eta = self.eta
            acq_eval = self.acq(X,eta=eta).numpy().reshape(num_x, self.acq.num_obj + self.acq.num_constr)
            #Get the results of the evaluation.
            out['F'] = acq_eval[:, :self.acq.num_obj]




class EvolutionOpt:
    def __init__(self,
            config_space ,
            acq          , 
            change_to_vector,
            **conf):
        self.config_space      = config_space
        self.acq        = acq
        self.pop        = conf.get('pop', 100)
        self.iter       = conf.get('iters',500)
        self.verbose    = False
        self.repair     = None
        self.sobol_init = True
        assert(self.acq.num_obj > 1)
        self.es = 'nsga2' 
        self.config_space_to_cont_vector = change_to_vector


    #Change this with Sobol. :)
    def get_init_pop(self, initial_suggest : pd.DataFrame = None) -> np.ndarray:
        
        assert initial_suggest != None
    
        #Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6)) 

        init_design_def_kwargs = {
            "cs": self.config_space,  # type: ignore[attr-defined] # noqa F821
            "traj_logger": None,
            "rng": seed,
            "ta_run_limit": None,  # type: ignore[attr-defined] # noqa F821
            "configs": None,
            "n_configs_x_params": 0,
            "max_config_fracs": 0.0,
            "init_budget": self.pop-1
            } 
        sobol  = SobolDesign(**init_design_def_kwargs)
        population = sobol._select_configurations() #self.config_space.sample_configuration(size=initial_config_size)
        if not isinstance(population, List):
            population = [population]
        # the population is maintained in a list-of-vector form where each ConfigSpace
        # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
        X_candidates = np.array([self.config_space_to_cont_vector(individual) for individual in population])
        
        initial_suggest = pd.DataFrame(self.config_space_to_cont_vector(initial_suggest).reshape(1,-1))
        X_candidates = pd.DataFrame(X_candidates)


        #if initial_suggest is not None:
        init_pop = pd.concat([initial_suggest, X_candidates], axis = 0)

        #Transform to 0-1 -- Probably ready from change_to_vector.
        #x, xe = self.space.transform(init_pop)
        #print(np.hstack([init_pop]))

        return np.hstack([init_pop])

    def get_mutation(self,types):
        mask = []
        for name in types:
            mask.append('real')

        mutation = MixedVariableMutation(mask, {
            'real' : get_mutation('real_pm', eta = 20), 
            'int'  : get_mutation('int_pm', eta = 20)
        })
        return mutation

    def get_crossover(self,types):
        mask = []
        for name in types:
            mask.append('real')

        crossover = MixedVariableCrossover(mask, {
            'real' : get_crossover('real_sbx', eta = 15, prob = 0.9), 
            'int'  : get_crossover('int_sbx', eta = 15, prob = 0.9)
        })
        return crossover


    def maximize(self, initial_suggest : pd.DataFrame = None, fix_input : dict = None, return_pop = False, eta= None) -> pd.DataFrame:
        

        assert eta != None

        #Get bounds
        #Use get bounds here. To get the lb and ub
        types, bounds = get_types(config_space=self.config_space)
        lb        = [i[0] for i in bounds]
        ub        = [i[1] for i in bounds]

        #Essentially this calls the acquisition function :)
        prob      = BOProblem(lb, ub, self.acq, self.config_space,eta = eta)


        #Create an initial population
        init_pop  = self.get_init_pop(initial_suggest)

        #Somethings about the types...
        mutation  = self.get_mutation(types)
        crossover = self.get_crossover(types)

        #Get the optimization algorithm.
        algo      = get_algorithm(self.es, pop_size = self.pop, sampling = init_pop, mutation = mutation, crossover = crossover, repair = self.repair)

        #Minimize the acquisition by using the specific algorithm.
        res       = minimize(prob, algo, ('n_gen', self.iter), verbose = self.verbose)

        #Some checks.
        if res.X is not None and not return_pop:
            opt_x = res.X.reshape(-1, len(lb)).astype(float)
        else:
            opt_x = np.array([p.X for p in res.pop]).astype(float)
        
        #Hm...
        self.res  = res
        
        return opt_x
