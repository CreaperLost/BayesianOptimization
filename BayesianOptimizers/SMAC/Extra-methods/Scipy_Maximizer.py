import numpy as np
from initial_design.sobol_design import SobolDesign
from typing import List, Optional, Tuple
from ConfigSpace import Configuration
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.optimize import minimize
import numpy as np



class Scipy_Maximizer():

    def __init__(self,objective_function,config_space,n_cand):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_samples: int
            Number of candidates that are samples
        """
        self.config_space = config_space
        self.n_cand = n_cand
        self.objective_function = objective_function

    def maximize(self,configspace_to_vector,eta):
        """
        Maximizes the given acquisition function. 

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        float
            The value of the acquisition function
        """

        #Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6)) 

        configs: list[tuple[float, Configuration]] = []

        #Wrapper in order to change to a negative shit.
        def func(x: np.ndarray) -> np.ndarray:
            assert self.objective_function is not None and self.objective_function.eta is not None
            # Probably not needed
            # [Configuration(self.config_space, vector=x)]
            
            return -self.objective_function(x).flatten()

        
        bounds_config = [(0, 1) for _ in range(len(self.config_space))]

        lower_bounds = [i[0] for i in bounds_config]
        upper_bounds = [i[1] for i in bounds_config]

        # Explore the parameter space more throughly
        x_seeds = np.random.RandomState(seed =seed).uniform(lower_bounds,upper_bounds,size=( 20, len(bounds_config)))


        self.objective_function.update(eta)

        #Initial Acquisition Value
        max_acq = None
        
        for x_try in x_seeds:
            #print(x_seeds,eta,self.objective_function(x_seeds).flatten())
            res = minimize(func,
                       x_try,
                       bounds=bounds_config,
                       method="L-BFGS-B")

            # See if success
            if not res.success:
                continue
        
            
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -np.squeeze(res.fun) >= max_acq:
                x_max = res.x
                max_acq = -np.squeeze(res.fun)

        # the acquisition value and the configuration of the max found by DE..
        
        y_max , x_max = max_acq,np.clip(x_max, lower_bounds, upper_bounds)

        #Find the point of X_candidates with maximum Acquisition function.
        #return X_candidates[y.argmax()],y.argmax()

        return x_max, y_max
