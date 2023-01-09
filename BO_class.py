from tqdm import tqdm
import numpy as np 
import pandas as pd

"""
Base Class for Bayesian Optimization


"""
class Bayesian_Optimization_base():

    #Initialize the object of Bayesian Optimization.
    def __init__(self,initial_configs,configuration_space,surrogate_model,acquisition_function,objective_function,restrictions,seed) -> None:
        self.initial_configs = initial_configs
        self.configuration_space = configuration_space
        self.surrogate_model = surrogate_model
        self.completed_configurations = None
        self.acquisition_function = acquisition_function
        self.restrictions = restrictions
        self.current_data = None
        self.objective_function = objective_function
        self.seed = seed
        self.timer = None
        self.iterations = None
        pass    

    def check_restriction(self):
        if 'time_budget' in self.restrictions.keys():
            self.timer = self.restrictions.get('time_budget')
        elif 'iteration_budget' in self.restrictions.keys():
            self.iterations = self.restrictions.get('iteration_budget')

        
    def run_initial_configurations(self):
        if self.initial_configs is not None:
            for configuration in self.initial_configs:
                self.initial_configs.remove(configuration)
                self.run_objective(configuration)
        else:
            # getting first few random values
            X_tried = self.ds.get_random_sample(self.config.initial_random)
            X_list, Y_list = self.runUserObjective(X_tried)

            # in case initial random results are invalid try different samples
            n_tries = 1
            while len(Y_list) < self.config.initial_random and n_tries < 3:
                X_tried2 = self.ds.get_random_sample(self.config.initial_random - len(Y_list))
                X_list2, Y_list2 = self.runUserObjective(X_tried2)
                X_tried2.extend(X_tried2)
                X_list = np.append(X_list, X_list2)
                Y_list = np.append(Y_list, Y_list2)
                n_tries += 1

            if len(Y_list) == 0:
                raise ValueError("No valid configuration found to initiate the Bayesian Optimizer")
        return X_list, Y_list, X_tried
        
        
        pass

    def turn_configuration_to_df_sample(self,configuration):
        pass
    def run_objective(self,configuration):
        y_score = self.objective_function(configuration)
        sample = self.turn_configuration_to_df_sample(configuration)
        sample.append(y_score)
        self.completed_configurations.add(configuration)
        pass

    def fit(self):
        # Run number of iterations
        # To add a new one for time-budget.
        self.run_initial_configurations()

        pbar = tqdm(range(self.iterations))
        #for iter in pbar:
            

        pass


    def predict(self,y_train):
        pass