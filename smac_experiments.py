import numpy as np
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space
import openml
import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.insert(0, '..')
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np
from BayesianOptimizers.Conditional_BayesianOptimization.smac_hpo import SMAC_HPO
from BayesianOptimizers.Conditional_BayesianOptimization.smac_instance_hpo import SMAC_Instance_HPO
from BayesianOptimizers.Conditional_BayesianOptimization.random_smac import Random_SMAC
from csv import writer
import time 


def run_benchmark_total(optimizers_used =[],bench_config={},save=True,load=True):
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

                start_time = time.time()
                if opt == 'SMAC':
                    Optimization=SMAC_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                             repo=repo,max_evals=max_evals,seed=seed,objective_function=objective_function,n_workers=1)                
                elif opt =='ROAR':
                    Optimization=Random_SMAC(configspace=configspace,config_dict=config_dict,task_id=task_id,
                             repo=repo,max_evals=max_evals,seed=seed,objective_function=objective_function,n_workers=1)
                elif opt == 'SMAC_Instance':
                    Optimization=SMAC_Instance_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                             repo=repo,max_evals=max_evals,seed=seed,objective_function=objective_function_per_fold)
                else: 
                    print(opt)
                    raise RuntimeError
                
                if load == False:
                    Optimization.run()
                else:
                    Optimization.load()
                
                m_time = time.time()-start_time
                print('Measured Total Time ',m_time)
                print('Total Time',np.cumsum(Optimization.total_time)[-1])
                print(Optimization.inc_score,Optimization.inc_config)
                
                

                config_per_group_directory=parse_directory([config_per_optimizer_directory,opt])
                
                #Change this.
                y_evaluations = Optimization.fX
                surrogate_time_evaluations = Optimization.surrogate_time
                objective_time_evaluations= Optimization.objective_time
                acquisition_time_evaluations = Optimization.acquisition_time
                total_time_evaluations = Optimization.total_time

                if save == True and load == False:
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
                        
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(surrogate_time_evaluations).to_csv( parse_directory([ surrogate_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(objective_time_evaluations).to_csv( parse_directory([ objective_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(acquisition_time_evaluations).to_csv( parse_directory([ acquisition_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

                    if 'SMAC' in opt or 'ROAR' in opt:
                        #Save configurations and y results for each group.
                        for group in Optimization.save_configuration:
                            Optimization.save_configuration[group].to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt == 'Multi_RF_Local':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                elif load == True:
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))







def get_openml_data(speed = None):
    # 2074 needs 15 hours for 3 seeds per optimizer.
    assert speed !=None
    if speed == 'fast':
        return [14954,11,3918,3917,3021,43,167141,9952]
    return [2074,9976,9910,167125]
    

def get_jad_data(speed = None):
    assert speed !=None
    if speed == 'fast':
        return [842,851,850,] #839, 847,1114
    #  on all seeds 
    return [843,883,866]
    6

if __name__ == '__main__':
    config_of_data = { 'Jad':{'data_ids':get_jad_data},
                        'OpenML': {'data_ids':get_openml_data}      }
    opt_list = ['SMAC' ] # ,,'Random_Search','RF_Local',] 'SMAC_Instance'
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