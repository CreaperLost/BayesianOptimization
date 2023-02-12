import os
import time
import json
import logging
import numpy as np
import tqdm
"""Inputs to add."""
from initial_design.initial_random_uniform import init_random_uniform
from BayesianOptimizers.bo_base import BO_Base
import logging
import george
import numpy as np
from pybnn.dngo import DNGO
from priors.default_priors import DefaultPrior
from models.wrapper_bohamiann import WrapperBohamiann
from models.gaussian_process import GaussianProcess
from models.gaussian_process_mcmc import GaussianProcessMCMC
from models.random_forest import RandomForest
from maximizers.scipy_optimizer import SciPyOptimizer
from maximizers.random_sampling import RandomSampling
from maximizers.differential_evolution import DifferentialEvolution
from acquisition_functions.ei import EI
from acquisition_functions.pi import PI
from acquisition_functions.log_ei import LogEI
from acquisition_functions.lcb import LCB
from acquisition_functions.marginalization import MarginalizationGPMCMC
from initial_design.initial_latin_hypercube import init_latin_hypercube_sampling

logger = logging.getLogger(__name__)

"""
Code adapted from Robo Package
https://github.com/automl/RoBO/blob/master/robo/solver/bayesian_optimization.py
"""

class Random_Forest_Robo(BO_Base):

    def __init__(self, objective_function, lower, upper,
                 initial_design=init_random_uniform,
                 initial_points=3,
                 output_path=None,
                 train_interval=1,
                 n_restarts=1,
                 rng=None,
                 num_iterations = 10):
        """
        Implementation of the standard Bayesian optimization loop that uses
        an acquisition function and a model to optimize a given objective_function.
        This module keeps track of additional information such as runtime,
        optimization overhead, evaluated points and saves the output
        in a json file.
        Parameters
        ----------
        acquisition_function: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        model: ModelObject
            Model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        objective_function:
            Function handle for the objective function
        output_path: string
            Specifies the path where the intermediate output after each iteration will be saved.
            If None no output will be saved to disk.
        initial_design: function
            Function that returns some points which will be evaluated before
            the Bayesian optimization loop is started. This allows to
            initialize the model.
        initial_points: int
            Defines the number of initial points that are evaluated before the
            actual Bayesian optimization.
        train_interval: int
            Specifies after how many iterations the model is retrained.
        n_restarts: int
            How often the incumbent estimation is repeated.
        rng: np.random.RandomState
            Random number generator
        """
        assert upper.shape[0] == lower.shape[0], "Dimension miss match"
        assert np.all(lower < upper), "Lower bound >= upper bound"
        assert initial_points <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

        
        self.model = RandomForest(rng=rng)
        self.acquisition_function = EI(self.model)
        self.maximize_func = RandomSampling(self.acquisition_function, lower, upper, rng=rng)
 
        self.start_time = time.time()
        self.initial_design = initial_design
        self.objective_function = objective_function
        self.max_iter = num_iterations
        #Here we start with empty data.
        self.X = None
        self.fX = None
        #Capture some of the information about time.
        self.time_func_evals = []
        self.time_overhead = []
        self.train_interval = train_interval
        self.lower = lower
        self.upper = upper
        self.output_path = output_path
        self.time_start = None
        #Save the incumbents here.
        self.incumbents = []
        self.incumbents_values = []
        self.n_restarts = n_restarts
        self.init_points = initial_points
        self.runtime = []



    """

    Get more information through this 
    ----------------------------------

    x_best, f_min = bo.optimize(num_iterations, X=X_init, y=Y_init)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = [x.tolist() for x in bo.X]
    results["y"] = [y for y in bo.y]
    return results
    
    """
    def optimize(self, X=None, y=None):
        """
        The main Bayesian optimization loop
        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points
        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        # Save the time where we start the Bayesian optimization procedure
        self.time_start = time.time()

        #Literally, if we got no data. then... run the initial design.
        if X is None and y is None:

            # Initial design
            X = []
            y = []

            start_time_overhead = time.time()
            #Samples initial points
            init = self.initial_design(self.lower,
                                       self.upper,
                                       self.init_points,
                                       rng=self.rng)
            time_overhead = (time.time() - start_time_overhead) / self.init_points

            for i, x in enumerate(init):

                logger.info("Evaluate: %s", x)

                start_time = time.time()
                new_y = self.objective_function(x)

                X.append(x)
                y.append(new_y)
                #Cost per evaluation of objective function.
                self.time_func_evals.append(time.time() - start_time)
                self.time_overhead.append(time_overhead)

                logger.info("Configuration achieved a performance of %f in %f seconds",
                            y[i], self.time_func_evals[i])

                # Use best point seen so far as incumbent
                best_idx = np.argmin(y)
                incumbent = X[best_idx]
                incumbent_value = y[best_idx]

                self.incumbents.append(incumbent.tolist())
                self.incumbents_values.append(incumbent_value)

                self.runtime.append(time.time() - self.start_time)

                if self.output_path is not None:
                    self.save_output(i)

            self.X = np.array(X)
            self.fX = np.array(y)
        else:
            self.X = X
            self.fX = y

        # Main Bayesian optimization loop
        # After you evaluated the first initi_points, continue with as many iterations are left.

        for it in range(self.init_points, self.max_iter):
            logger.info("Start iteration %d ... ", it)
            if it % 10 == 0:
                print('Iter',it)
            start_time = time.time()


            #So the idea here is that you may don't want to retrain the acquisition all the time
            #Alway re-optimize.
            if it % self.train_interval == 0:
                do_optimize = True
            else:
                do_optimize = False

            # Choose next point to evaluate
            #new_x = self.choose_next(self.X, self.fX, do_optimize)
            #Experimental
            #Train the Model
            try:
                logger.info("Train model...")
                t = time.time()
                self.model.train(self.X, self.fX, do_optimize=do_optimize)
                logger.info("Time to train the model: %f", (time.time() - t))
            except:
                logger.error("Model could not be trained!")
                raise

            #Run acquisition function.
            self.acquisition_function.update(self.model)

            logger.info("Maximize acquisition function...")
            t = time.time()
            #Maximize acquisition.
            new_x = self.maximize_func.maximize()

            logger.info("Time to maximize the acquisition function: %f", (time.time() - t))
            

            self.time_overhead.append(time.time() - start_time)
            logger.info("Optimization overhead was %f seconds", self.time_overhead[-1])
            logger.info("Next candidate %s", str(new_x))

            # Evaluate the objective function on the new point
            start_time = time.time()
            new_y = self.objective_function(new_x)
            self.time_func_evals.append(time.time() - start_time)

            logger.info("Configuration achieved a performance of %f ", new_y)
            logger.info("Evaluation of this configuration took %f seconds", self.time_func_evals[-1])

            # Extend the data
            self.X = np.append(self.X, new_x[None, :], axis=0)
            self.fX = np.append(self.fX, new_y)

            # Estimate incumbent
            best_idx = np.argmin(self.fX)
            incumbent = self.X[best_idx]
            incumbent_value = self.fX[best_idx]

            self.incumbents.append(incumbent.tolist())
            self.incumbents_values.append(incumbent_value)
            logger.info("Current incumbent %s with estimated performance %f",
                        str(incumbent), incumbent_value)

            self.runtime.append(time.time() - self.start_time)

            if self.output_path is not None:
                self.save_output(it)

        logger.info("Return %s as incumbent with error %f ",
                    self.incumbents[-1], self.incumbents_values[-1])

        return self.incumbents[-1], self.incumbents_values[-1]

    def choose_next(self, X=None, y=None, do_optimize=True):
        """
        Suggests a new point to evaluate. and trains the surrogate model as well...Should change in the future.
        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points
        do_optimize: bool
            If true the hyperparameters of the model are
            optimized before the acquisition function is
            maximized.
        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """

        if X is None and y is None:
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        elif X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        else:
            try:
                logger.info("Train model...")
                t = time.time()
                self.model.train(X, y, do_optimize=do_optimize)
                logger.info("Time to train the model: %f", (time.time() - t))
            except:
                logger.error("Model could not be trained!")
                raise
            self.acquisition_function.update(self.model)

            logger.info("Maximize acquisition function...")
            t = time.time()
            x = self.maximize_func.maximize()

            logger.info("Time to maximize the acquisition function: %f", (time.time() - t))

        return x

    def save_output(self, it):

        data = dict()
        data["optimization_overhead"] = self.time_overhead[it]
        data["runtime"] = self.runtime[it]
        data["incumbent"] = self.incumbents[it]
        data["incumbents_value"] = self.incumbents_values[it]
        data["time_func_eval"] = self.time_func_evals[it]
        data["iteration"] = it

        json.dump(data, open(os.path.join(self.output_path, "robo_iter_%d.json" % it), "w"))