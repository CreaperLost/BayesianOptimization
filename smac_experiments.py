from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space
import openml
import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
from pathlib import Path
from BayesianOptimizers.Conditional_BayesianOptimization.Group_Random_Search import Group_Random_Search
from BayesianOptimizers.Conditional_BayesianOptimization.Group_SMAC_base import Group_Bayesian_Optimization
from BayesianOptimizers.Conditional_BayesianOptimization.MultiFold_Group_Smac_base import MultiFold_Group_Bayesian_Optimization

import os
import sys
sys.path.insert(0, '..')
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np
from smac import MultiFidelityFacade as MFFacade



def run_benchmark_total(optimizers_used =[],bench_config={},save=True):
    assert optimizers_used != []
    assert bench_config != {}

    #Optimizer related
    n_init = bench_config['n_init']
    max_evals = bench_config['max_evals']

    #Dataset related
    n_datasets = bench_config['n_datasets']
    data_ids  = bench_config['data_ids']
    n_seeds  = bench_config['n_seeds']

    data_repo  = bench_config['data_repo']

    #Benchmark related fields
    type_of_bench = bench_config['type_of_bench'] 
    benchmark_name = bench_config['bench_name']
    benchmark_class = bench_config['bench_class']
    

    optimizers_list = optimizers_used

    assert optimizers_list != [] or optimizers_list != None


    main_directory = os.getcwd()

    #Directory to save results for specific metric
    score_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Metric'])

    #Directory to save time for the optimizers
    surrogate_time_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Surrogate_Time'])
    #Directory to save time for the optimizers
    objective_time_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Objective_Time'])
    #Directory to save time for the optimizers
    acquisition_time_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Acquisition_Time'])
    #Directory to save time for the optimizers
    total_time_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Total_Time'])

    #Directory to save configurations per group
    config_directory = parse_directory([main_directory,type_of_bench,benchmark_name,'Configurations'])


    curr_dataset = 0

    #Add the type of Dataset repo before.
    score_directory = parse_directory([score_directory,data_repo])
    surrogate_time_directory = parse_directory([surrogate_time_directory,data_repo])
    objective_time_directory = parse_directory([objective_time_directory,data_repo])
    acquisition_time_directory = parse_directory([acquisition_time_directory,data_repo])
    total_time_directory = parse_directory([total_time_directory,data_repo])
    config_directory = parse_directory([config_directory,data_repo])

    for task_id in data_ids:
        
        #These Folders will have seed results in them.
        score_per_seed_directory = parse_directory([score_directory,'Dataset' +str(task_id)])
        surrogate_time_per_seed_directory  = parse_directory([surrogate_time_directory, 'Dataset' +str(task_id)])
        objective_time_per_seed_directory  = parse_directory([objective_time_directory, 'Dataset' +str(task_id)])
        acquisition_time_per_seed_directory  = parse_directory([acquisition_time_directory, 'Dataset' +str(task_id)])
        total_time_per_seed_directory  = parse_directory([total_time_directory, 'Dataset' +str(task_id)])
        config_per_seed_directory = parse_directory([config_directory,'Dataset' +str(task_id)])



        for seed in n_seeds:

            score_per_optimizer_directory = parse_directory([score_per_seed_directory,'Seed' + str(seed) ])
            surrogate_time_per_optimizer_directory = parse_directory([surrogate_time_per_seed_directory,'Seed' + str(seed) ])
            objective_time_per_optimizer_directory = parse_directory([objective_time_per_seed_directory,'Seed' + str(seed) ])
            acquisition_time_per_optimizer_directory = parse_directory([acquisition_time_per_seed_directory,'Seed' + str(seed) ])
            total_time_per_optimizer_directory = parse_directory([total_time_per_seed_directory,'Seed' + str(seed) ])
            
            config_per_optimizer_directory = parse_directory([config_per_seed_directory,'Seed' + str(seed) ])


            for opt in optimizers_list: 
                
                benchmark_ = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo)
                #Get the config Space
                configspace,config_dict = benchmark_.get_configuration_space()

                #Get the benchmark.
                objective_function = benchmark_.smac_objective_function
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.smac_objective_function_per_fold

                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                # Scenario object specifying the optimization environment
                scenario = Scenario(configspace,name='Smac on dataset '+str(task_id) + 'seed ' + str(seed), n_trials=max_evals,deterministic=True,seed=seed)

                # Use SMAC to find the best configuration/hyperparameters
                smac = HyperparameterOptimizationFacade(scenario, objective_function,seed=seed)
                incumbent = smac.optimize()
                print(incumbent)


                scenario = Scenario(
                        configspace,
                        n_trials=550,  # We want to try max 5000 different trials
                        min_budget=1,  # Use min 1 fold.
                        max_budget=10,  # Use max 10 folds. 
                        instances=[0,1,2,3,4,5,6,7,8,9],
                    )
                # Create our SMAC object and pass the scenario and the train method
                smac = MFFacade(
                    scenario,
                    objective_function_per_fold,
                    overwrite=True,
                )

                # Now we start the optimization process
                incumbent = smac.optimize()
                print(incumbent)

                
                #The file path for current optimizer.
                config_per_group_directory=parse_directory([config_per_optimizer_directory,opt])
                
                
                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(surrogate_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(objective_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(acquisition_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(total_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(config_per_group_directory).mkdir(parents=True, exist_ok=True)
                    except FileExistsError:
                        print("Folder is already there")
                        
                    else:
                        print("Folder is created there")
                        
                    """pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(surrogate_time_evaluations).to_csv( parse_directory([ surrogate_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(objective_time_evaluations).to_csv( parse_directory([ objective_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(acquisition_time_evaluations).to_csv( parse_directory([ acquisition_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))
                    """








def get_openml_data(speed = None):
    # 2074 needs 15 hours for 3 seeds per optimizer.
    assert speed !=None
    if speed == 'fast':
        return [14954,11,3918,3917,3021,43,167141,9952]
    return [2074,9976,9910,167125]
    

def get_jad_data(speed = None):
    assert speed !=None
    if speed == 'fast':
        return [839, 847,1114] #842,851,850
    #  on all seeds 
    return [843,883,866]
    

if __name__ == '__main__':
    config_of_data = { 'Jad':{'data_ids':get_jad_data},
                        'OpenML': {'data_ids':get_openml_data}      }
    opt_list = ['SMAC' ] # ,,'Random_Search','RF_Local',]
    for speed in ['fast']:
     # obtain the benchmark suite    
        for repo in ['Jad','OpenML']:
            #XGBoost Benchmark    
            xgb_bench_config =  {
                'n_init' : 10,
                'max_evals' : 550,
                'n_datasets' : 1000,
                'data_ids' :  config_of_data[repo]['data_ids'](speed=speed),
                'n_seeds' : [1,2,3], #
                'type_of_bench': 'Main_Multi_Fold_Group_Space_Results',
                'bench_name' :'GROUP',
                'bench_class' : Group_MultiFold_Space,
                'data_repo' : repo
            }
            run_benchmark_total(opt_list,xgb_bench_config)