from smac import MultiFidelityFacade as MFFacade
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
from smac.facade.random_facade import RandomFacade

class Random_SMAC:
                

    def __init__(self,configspace,config_dict,task_id,repo,max_evals,seed,objective_function,n_workers=1):
        #Scenario object specifying the optimization environment
        self.scenario = Scenario(configspace,name='Dataset'+str(task_id),
                                           output_directory='SIngle_smac_random' +str(n_workers) +'/'+repo,
                                     n_trials=max_evals,deterministic=True,seed=seed,n_workers = n_workers )
         
        # Use SMAC to find the best configuration/hyperparameters
        self.configspace = configspace
        self.save_configuration={}
        for i in config_dict:
            self.save_configuration[i] = pd.DataFrame()
        self.objective_function = objective_function

        
        self.fX = np.array([])
        self.X_group = np.array([])
        self.total_time = np.array([])

        self.acquisition_time = np.array([0 for i in range(max_evals)])
        self.surrogate_time = np.array([0 for i in range(max_evals)])
        self.objective_time = np.array([0 for i in range(max_evals)])
        
        self.inc_score = np.inf
        self.inc_config = {}

    def run(self):

        smac_single = RandomFacade(self.scenario, self.objective_function,overwrite=True)
        incumbent_single = smac_single.optimize()
        
        # Plot all trials
        for trial_info, trial_value in smac_single.runhistory.items():
                    
            # Trial info
            config_descr = smac_single.runhistory.get_config(trial_info.config_id)
                
            # Trial value
            cost = trial_value.cost
            time = trial_value.time

            if cost < self.inc_score:
                self.inc_config  = config_descr.get_dictionary()
                self.inc_score = cost
                print(f'Best config {self.inc_config} at step: {trial_info.config_id}, score :{cost}')

            #add time and fX.
            self.fX = np.append(self.fX,np.array(cost))
            self.total_time = np.append(self.total_time,np.array(time))   

            self.X_group = np.append(self.X_group,config_descr['model']) 
            #This is a better interpretable form of storing the configurations.
            new_config = config_descr.get_dictionary().copy()
            new_config.pop('model') 

            X_df = pd.DataFrame(new_config,index=[0])
            y_df = pd.DataFrame({'y':cost},index=[0])
            new_row = pd.concat([X_df,y_df],axis=1)
            self.save_configuration[config_descr['model']] = self.save_configuration[config_descr['model']].append(new_row,ignore_index=True)
            

            