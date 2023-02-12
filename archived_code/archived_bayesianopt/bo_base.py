
import os
import csv
import time
import errno
import logging
import json

logger = logging.getLogger(__name__)
"""
Code taken from ROBO Package:
https://github.com/automl/RoBO/blob/master/robo/solver/base_solver.py
https://numairmansur.github.io/RoboDocumentation1/
"""
class BO_Base(object):

    def __init__(self, acquisition_function=None, surrogate_model=None,
                 maximize_function=None, task=None, save_dir=None):
        """
        Base Class of BO optimization
        Parameters
        ----------
        acquisition_function: BaseAcquisitionFunction Object
            The acquisition function which will be maximized.
        surrogate_model: surrogate_modelObject
            surrogate_model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        maximize_function: MaximizerObject
            Optimization method that is used to maximize the acquisition
            function
        save_dir: String
            Output path
        """

        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.maximize_function = maximize_function
        self.task = task
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.create_save_dir()

    def create_save_dir(self):
        """
        Creates the save directory to store the runs
        """
        try:
            os.makedirs(self.save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise 
        self.output_file = open(os.path.join(self.save_dir, 'results.csv'), 'w')
        self.output_file_json = open(os.path.join(self.save_dir, 'results.json'), 'w')
        self.csv_writer = None
        self.json_writer = None

    def get_observations(self):
        return self.X, self.Y

    def get_surrogate_model(self):
        if self.surrogate_model is None:
            logger.info("No surrogate_model trained yet!")
        return self.surrogate_model

    def run(self, num_iterations=10, X=None, y=None):
        """
        The main optimization loop
        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,)
            Function values of the already evaluated points
        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        pass

    def choose_next(self, X=None, y=None):
        """
        Suggests a new point to evaluate.
        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,)
            Function values of the already evaluated points
        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """
        pass

    def get_json_data(self, it):
        """
        Json getter function
        :return: dict() object
        """

        jsonData = {"optimization_overhead":self.time_overhead[it], "runtime": time.time() - self.time_start,
                    "incumbent": self.incumbent.tolist(),
                    "incumbent_fval": self.incumbent_value.tolist(),
                    "time_func_eval": self.time_func_eval[it],
                    "iteration": it}
        return jsonData

    def save_json(self, it, **kwargs):
        """
        Saves meta information of an iteration in a Json file.
        """
        base_solver_data =self.get_json_data(it)
        base_surrogate_model_data = self.surrogate_model.get_json_data()
        base_task_data = self.task.get_json_data()
        base_acquisition_data = self.acquisition_function.get_json_data()

        data = {'Solver': base_solver_data,
                'surrogate_model': base_surrogate_model_data,
                'Task': base_task_data,
                'Acquisiton': base_acquisition_data
                }

        json.dump(data, self.output_file_json)
        self.output_file_json.write('\n')  # Json more readable. Drop it?