from smac import MultiFidelityFacade as MFFacade
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
from smac.intensifier.intensifier import Intensifier
import time 

class SMAC_Instance_HPO:
                 

    def __init__(self,configspace,config_dict,task_id,repo,max_evals,seed,objective_function):
        

        # Use SMAC to find the best configuration/hyperparameters
        self.configspace = configspace
        self.save_configuration={}
        for i in config_dict:
            self.save_configuration[i] = pd.DataFrame()
        self.objective_function = objective_function

        instances = [0,1,2,3,4]
        instance_features = dict()
        for i in instances:
            instance_features[i] = [i]
        
        self.scenario = Scenario(configspace,name='Dataset'+str(task_id),
                            output_directory = 'per_instance_smac/'+repo,
                            n_trials=max_evals,  # We want to try max 5000 different trials
                            min_budget=1,max_budget=5,  # Use min 1 fold, Use max 10 folds. 
                            instances=instances,
                            instance_features = instance_features,
                            deterministic=True,seed=seed)
        
        self.intensifier = Intensifier(scenario=self.scenario, max_config_calls=len(instances),seed=seed)
        

        

        self.per_instance_smac = HyperparameterOptimizationFacade(
                    scenario= self.scenario,
                    target_function = self.objective_function,
                    intensifier=self.intensifier,
                    overwrite=True,
                )


        

        self.max_evals = max_evals
        self.fX = np.array([])
        self.X_group = np.array([])

        self.inc_score = np.inf
        self.inc_config = {}

        self.total_time = pd.DataFrame(columns=['Time','Score'])  

    def run(self):
        
        start_time = time.time()
    
        incumbent_per_instance = self.per_instance_smac.optimize()
        
        end_time = time.time() - start_time

        ids_already_validated = []

        """# Plot all trials
        for trial_info, trial_value in self.per_instance_smac.runhistory.items():
            
            #The unique id of the configuration
            config_id = trial_info.config_id
            
            # Here we get the configuration.
            config_descr = self.per_instance_smac.runhistory.get_config(config_id)
                
            if config_id in ids_already_validated:
                continue
            
            # Run the proposed configuration on all folds.
            cost = self.per_instance_smac.validate(config=config_descr)
            ids_already_validated.append(config_id)

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


        """

        print(f'Budget used : {len(self.per_instance_smac.runhistory.items())} out of {self.max_evals}')
        configs_run = []
        for item in self.per_instance_smac.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            #print(item)

            config_descr = self.per_instance_smac.runhistory.get_config(item.config_ids[0])
            #print(config_descr)
            cost = self.per_instance_smac.validate(config=config_descr)
            #print(item.costs[0],cost)
            for i,(trial_info, trial_value) in enumerate(self.per_instance_smac.runhistory.items()):
                config_id = trial_info.config_id
                if config_id == item.config_ids[0]:
                    configs_run.append(i)
                    break

            y = cost #self.fX[ids_already_validated.index(item.config_ids[0])]
            x = item.walltime
            new_row = pd.DataFrame({'Time':x,'Score':y},index=[0])
            self.total_time = self.total_time.append(new_row,ignore_index=True)


            new_config = config_descr.get_dictionary().copy()
            new_config.pop('model') 
            X_df = pd.DataFrame(new_config,index=[0])
            y_df = pd.DataFrame({'y':cost},index=[0])
            new_row = pd.concat([X_df,y_df],axis=1)
            self.save_configuration[config_descr['model']] = self.save_configuration[config_descr['model']].append(new_row,ignore_index=True)


            # final_scores kept here.
            self.fX = np.append(self.fX,np.array(cost))

        #Finally append the final score!!!!!! YEAS
        #print(f'Score of supposed inc : {y} equals actual inc {self.inc_score}')
        new_row = pd.DataFrame({'Time':end_time,'Score':self.inc_score},index=[0])
        
        self.total_time = self.total_time.append(new_row,ignore_index=True)
        #print(self.fX)
        #print(configs_run)
        new_fx = [100 for i in range(self.max_evals)]
        prev = 0
        for idx,pos in enumerate(configs_run):
            score = self.fX[idx]
            for idx_n in range(pos,self.max_evals):
                new_fx[idx_n] = score 
            prev = pos+1
        self.fX = np.array(new_fx)
        print(self.max_evals, len(self.fX))
        print(self.fX)