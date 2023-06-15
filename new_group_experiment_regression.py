import numpy as np
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
from benchmarks.Group_MulltiFoldBenchmark_Regression import Group_MultiFold_Space_Regression
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np
"""

"""
from BayesianOptimizers.Conditional_BayesianOptimization.Group_Random_Search import Group_Random_Search
from BayesianOptimizers.Experimental.Pavlos_BO import Pavlos_BO
from BayesianOptimizers.Experimental.PavlosV2 import PavlosV2
from BayesianOptimizers.Conditional_BayesianOptimization.MultiFold_Group_Smac_base import MultiFold_Group_Bayesian_Optimization
from BayesianOptimizers.Conditional_BayesianOptimization.Progressive_group_smac import Progressive_BO
from BayesianOptimizers.Conditional_BayesianOptimization.smac_hpo import SMAC_HPO

from csv import writer
import time 


def run_benchmark_total(optimizers_used =[],bench_config={},save=True):
    assert optimizers_used != []
    assert bench_config != {}

    #Optimizer related
    n_init,max_evals = bench_config['n_init'],bench_config['max_evals']

    #Dataset related
    data_ids ,n_seeds = bench_config['data_ids'],bench_config['n_seeds']

    data_repo  = bench_config['data_repo']

    #Benchmark related fields
    type_of_bench = bench_config['type_of_bench'] 
    benchmark_name = bench_config['bench_name']
    benchmark_class = bench_config['bench_class']
    
    optimizers_list = optimizers_used

    assert optimizers_list != [] or optimizers_list != None


    main_directory = os.getcwd()
    
    
    for task_id in data_ids:
        task_id_str = 'Dataset' +str(task_id)
        for seed in n_seeds:
            Seed_id_str = 'Seed' + str(seed)
            for opt in optimizers_list: 
            
                score_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Metric',data_repo,task_id_str,Seed_id_str,opt)
                total_time_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Total_Time',data_repo,task_id_str,Seed_id_str,opt)
                config_per_group_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Configurations',data_repo,task_id_str,Seed_id_str,opt)

                benchmark_ = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo)
                #Get the config Space
                configspace,config_dict = benchmark_.get_configuration_space()

                #Get the benchmark.
                smac_objective_function = benchmark_.smac_objective_function
                
                #Get the objective_function per fold.
                smac_objective_function_per_fold = benchmark_.smac_objective_function_per_fold

                #Get the benchmark.
                objective_function = benchmark_.objective_function
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.objective_function_per_fold

                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                
                if opt == 'Random_Search':
                    Optimization = Group_Random_Search(f=objective_function,configuration_space= configspace,n_init = n_init,max_evals= max_evals,random_seed=seed)
                elif opt == 'Pavlos':
                    Optimization = Pavlos_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Multi_RF_Local':
                    Optimization = MultiFold_Group_Bayesian_Optimization(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'PavlosV2':
                    Optimization = PavlosV2(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Progressive_BO':
                    Optimization = Progressive_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'SMAC':
                    Optimization = SMAC_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                    repo=data_repo,max_evals=max_evals,seed=seed,objective_function=smac_objective_function,n_workers=1)  
                else: 
                    print(opt)
                    raise RuntimeError
                """            
                 
                elif opt == 'SMAC_Instance':
                    Optimization= SMAC_Instance_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                             repo=repo,max_evals=2*max_evals,seed=seed,objective_function=smac_objective_function_per_fold)
                """
                
                
                
                
                start_time = time.time()
                Optimization.run()
                m_time = time.time()-start_time
                print('Measured Total Time ',m_time)
                
                
                #Change this.
                y_evaluations = Optimization.fX
                total_time_evaluations = Optimization.total_time

                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(total_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(config_per_group_directory).mkdir(parents=True, exist_ok=True)
                    except FileExistsError:
                        print("Folder is already there")
                    else:
                        print("Folder is created there")
                        
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

                    if 'SMAC' in opt or 'ROAR' in opt:
                        #Save configurations and y results for each group.
                        for group in Optimization.save_configuration:
                            Optimization.save_configuration[group].to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt == 'Multi_RF_Local' or opt == 'Progressive_BO':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt =='Random_Search':
                        #Save configurations and y results for each group.
                        for group in Optimization.fX_per_group:
                            X_df = Optimization.X_per_group[group]
                            y_df = pd.DataFrame({'y':Optimization.fX_per_group[group]})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                    elif opt == 'Pavlos' or opt == 'PavlosV2':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    
    






def get_openml_data(speed = None):
    # 2074 needs 15 hours for 3 seeds per optimizer.
    assert speed !=None


    """
    Problematic:
    1. 361618 : This dataset has no categorical indexes but has categorical attributes!
    
    
    """


    if speed == 'fast': 
        # 359933, 359944, 233215, 167210, 359949, 359936, 
        # 359946, 359938, 359939, 359940, 359942, 359935, 
        # 360933, 360932, 233214, 359948, 359930, 359945,
        # 359951, 359945, 360945, 359932, 359931, 359950, 359934,359948
        """l = [359949,359936,359946,359938,359939,359940,\
             359942,359935,360933,360932,233214,,\
             ]"""
        l.reverse()
        return l
    #These are from new benchmark
    return [361619,361617,361259,361258,361256,361249]

def get_jad_data(speed = None):        
    assert speed !=None
    if speed == 'fast':
        return []
    return [] 

if __name__ == '__main__':
    config_of_data = { 'Jad':{'data_ids':get_jad_data},
                        'OpenML': {'data_ids':get_openml_data}      }
    
    #9976 datasets hasn't run for SMAC.

    opt_list = ['SMAC','Pavlos','Random_Search','Multi_RF_Local','Progressive_BO'] # ,,,'RF_Local',] 'SMAC_Instance' 'SMAC' ,'SMAC' ,,, 'Pavlos','Random_Search','Multi_RF_Local' 'SMAC', 'Pavlos','Random_Search','Multi_RF_Local','Random_Search','Multi_RF_Local','Pavlos'
    for speed in ['fast']: 
     # obtain the benchmark suite    
        for repo in ['OpenML']: #'
            #XGBoost Benchmark    
            xgb_bench_config =  {
                'n_init' : 10,
                'max_evals' : 1050,
                'n_datasets' : 1000,
                'data_ids' :  config_of_data[repo]['data_ids'](speed=speed),
                'n_seeds' : [1], 
                'type_of_bench': 'Main_Multi_Fold_Group_Space_Results_Rregression',
                'bench_name' :'GROUP',
                'bench_class' : Group_MultiFold_Space_Regression,
                'data_repo' : repo
            }
            run_benchmark_total(opt_list,xgb_bench_config)
