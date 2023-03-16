import numpy as np
from initial_design.sobol_design import SobolDesign
from typing import List, Optional, Tuple
from ConfigSpace import Configuration
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
 
class DE_Maximizer():

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
            assert self.objective_function is not None
            # Probably not needed
            # [Configuration(self.config_space, vector=x)]
            return -self.objective_function(x).flatten()

        #Update first.
        self.objective_function.update(eta)

        #Initial Acquisition Value
        max_acq = None

        # Set  some bounds...
        ds = DifferentialEvolutionSolver(
            func,
            bounds=[[0, 1] for _ in range(len(self.config_space))],
            args=(),
            strategy="best1bin",
            maxiter=20,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=seed,
            polish=True,
            callback=None,
            updating='immediate',
            disp=False,
            init="latinhypercube",
            atol=0
        )

        _ = ds.solve()
        # for each of the pop. insert into a list.
        for pop, val in zip(ds.population, ds.population_energies):
            #rc = Configuration(self.config_space, vector=pop)
            rc = pop
            # save the actual expected improvement.
            configs.append((-val, rc))

        #sort in ascending order (lower acquisition values first)
        configs.sort(key=lambda t: t[0])
        #Higher acquisition now as we reverse to descending order
        configs.reverse()

        
        print(configs[0])
        # the acquisition value and the configuration of the max found by DE..
        y_max , x_max = configs[0]

        

        #Find the point of X_candidates with maximum Acquisition function.
        #return X_candidates[y.argmax()],y.argmax()

        return x_max, y_max
