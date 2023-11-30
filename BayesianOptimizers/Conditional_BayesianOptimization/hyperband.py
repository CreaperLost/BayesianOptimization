from smac import MultiFidelityFacade as MFFacade
from smac import HyperparameterOptimizationFacade, Scenario, HyperbandFacade
from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
import time
from smac.intensifier.hyperband import Hyperband

class HyperBand:
                

    def __init__(self,configspace,config_dict,task_id,repo,max_evals,seed,objective_function,n_workers=1,init_evals=None,min_budget= 0.5, max_budget = 1):

        #Scenario object specifying the optimization environment
        self.scenario = Scenario(configspace,name='Dataset'+str(task_id),
                                           output_directory='single_smac' +'/'+repo,
                                     n_trials=max_evals,deterministic=True,seed=seed,n_workers = n_workers, min_budget=1,max_budget=9 )
        self.initial_design = None

        self.seed =seed
        
        # Use SMAC to find the best configuration/hyperparameters
        self.configspace = configspace
        self.save_configuration={}
        for i in config_dict:
            self.save_configuration[i] = pd.DataFrame()
        self.objective_function = objective_function

        
        self.fX = np.array([])
        self.X_group = np.array([])
        

        self.acquisition_time = np.array([0 for i in range(max_evals)])
        self.surrogate_time = np.array([0 for i in range(max_evals)])
        self.objective_time = np.array([0 for i in range(max_evals)])
        
        self.inc_score = np.inf
        self.inc_config = {}

        self.total_time = pd.DataFrame(columns=['Time','Score'])  
        

    
    def run(self):
        start_time = time.time()


        custom_intensifier = Hyperband(self.scenario,incumbent_selection='any_budget')

        hyper_band = HyperbandFacade(self.scenario, self.objective_function,overwrite=True,intensifier=custom_intensifier)
        incumbent_single = hyper_band.optimize()
        

        #print('Run History : ',len(hyper_band.runhistory.items()))

        # Plot all trials
        for trial_info, trial_value in hyper_band.runhistory.items():
                    
            # Trial info
            config_descr = hyper_band.runhistory.get_config(trial_info.config_id)
                
            # Trial value
            cost = trial_value.cost
            #time = trial_value.time

            if cost < self.inc_score:
                self.inc_config  = config_descr.get_dictionary()
                self.inc_score = cost
                print(f'Best config {self.inc_config} at step: {trial_info.config_id}, score :{cost}')

            #add time and fX.
            self.fX = np.append(self.fX,np.array(cost))
            
            #self.total_time = np.append(self.total_time,np.array(time))   

            self.X_group = np.append(self.X_group,config_descr['model']) 
            #This is a better interpretable form of storing the configurations.
            new_config = config_descr.get_dictionary().copy()
            new_config.pop('model') 

            X_df = pd.DataFrame(new_config,index=[0])
            y_df = pd.DataFrame({'y':cost},index=[0])
            new_row = pd.concat([X_df,y_df],axis=1)
            self.save_configuration[config_descr['model']] = self.save_configuration[config_descr['model']].append(new_row,ignore_index=True)

        #print('Trajectory : ',len(hyper_band.intensifier.trajectory))
        
        for item in hyper_band.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1
            config_descr = hyper_band.runhistory.get_config(item.config_ids[0])
            
            y = item.costs[0]
            x = item.walltime
            new_row = pd.DataFrame({'Time':x,'Score':y},index=[0])
            self.total_time = self.total_time.append(new_row,ignore_index=True)

        #Finally append the final score!!!!!! YEAS
        end_time = time.time() - start_time
        #print(f'this { hyper_band._get_optimizer().used_walltime } == {end_time}')
        new_row = pd.DataFrame({'Time':end_time,'Score':y},index=[0])


        print(np.min(self.fX),new_row.iloc[-1])
        
        self.total_time = self.total_time.append(new_row,ignore_index=True)

        print(incumbent_single,self.inc_config)
            