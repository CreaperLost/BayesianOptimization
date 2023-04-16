import openml
import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
from pathlib import Path
from BayesianOptimizers.SMAC.Random_Search import Random_Search
from BayesianOptimizers.SMAC.smac_base import Bayesian_Optimization
from BayesianOptimizers.SMAC.smac_base_multifold import Bayesian_Optimization_MultiFold
import os
import sys
sys.path.insert(0, '..')
from benchmarks.MultiFold_XGBoostBenchmark import MultiFold_XGBoostBenchmark
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path




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

    curr_dataset = 0

    #Add the type of Dataset repo before.
    score_directory = parse_directory([score_directory,data_repo])
    surrogate_time_directory = parse_directory([surrogate_time_directory,data_repo])
    objective_time_directory = parse_directory([objective_time_directory,data_repo])
    acquisition_time_directory = parse_directory([acquisition_time_directory,data_repo])
    total_time_directory = parse_directory([total_time_directory,data_repo])


    for task_id in data_ids:
        
        #These Folders will have seed results in them.
        score_per_seed_directory = parse_directory([score_directory,'Dataset' +str(task_id)])
        surrogate_time_per_seed_directory  = parse_directory([surrogate_time_directory, 'Dataset' +str(task_id)])
        objective_time_per_seed_directory  = parse_directory([objective_time_directory, 'Dataset' +str(task_id)])
        acquisition_time_per_seed_directory  = parse_directory([acquisition_time_directory, 'Dataset' +str(task_id)])
        total_time_per_seed_directory  = parse_directory([total_time_directory, 'Dataset' +str(task_id)])

        for seed in n_seeds:

            score_per_optimizer_directory = parse_directory([score_per_seed_directory,'Seed' + str(seed) ])
            surrogate_time_per_optimizer_directory = parse_directory([surrogate_time_per_seed_directory,'Seed' + str(seed) ])
            objective_time_per_optimizer_directory = parse_directory([objective_time_per_seed_directory,'Seed' + str(seed) ])
            acquisition_time_per_optimizer_directory = parse_directory([acquisition_time_per_seed_directory,'Seed' + str(seed) ])
            total_time_per_optimizer_directory = parse_directory([total_time_per_seed_directory,'Seed' + str(seed) ])

            for opt in optimizers_list: 
                
                benchmark_ = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo)
                #Get the config Space
                configspace = benchmark_.get_configuration_space()

                #Get the benchmark.
                objective_function = benchmark_.objective_function
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.objective_function_per_fold

                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                if opt == 'Random_Search':
                    Optimization = Random_Search(f=objective_function,configuration_space= configspace,n_init = n_init,max_evals= max_evals,random_seed=seed)
                # Single k-Fold Validation.
                elif opt == 'RF_Local':
                    Optimization = Bayesian_Optimization(f=objective_function, model='RF' ,lb= None, ub =None , configuration_space= configspace ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local')
                elif opt == 'Multi_RF_Local':
                    Optimization = Bayesian_Optimization_MultiFold(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= configspace ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local')
                else: 
                    print(opt)
                    raise RuntimeError
 

                best_score = Optimization.run()

                #Get the evaluatinons.
                y_evaluations = Optimization.fX

                #Change this.
                surrogate_time_evaluations = Optimization.surrogate_time
                objective_time_evaluations= Optimization.objective_time
                acquisition_time_evaluations = Optimization.acquisition_time
                total_time_evaluations = Optimization.total_time
                

                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(surrogate_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(objective_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(acquisition_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(total_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                    except FileExistsError:
                        print("Folder is already there")
                        """"""
                    else:
                        print("Folder is created there")
                        """"""
                        #print("Folder was created")
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(surrogate_time_evaluations).to_csv( parse_directory([ surrogate_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(objective_time_evaluations).to_csv( parse_directory([ objective_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(acquisition_time_evaluations).to_csv( parse_directory([ acquisition_time_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

        # Just in case we want less.
        curr_dataset+=1
        if curr_dataset >= n_datasets:
            break 

def get_openml_data():
    tasks = [11,14954,43,3021,3917,3918,9952,167141,2074,9910] #,167125,9976
    #tasks = [167125,9976]
    return tasks


#
def get_jad_data():
    interesting_Data = [851,842,1114,839,847,850,843] #,866, 883, 
    #interesting_Data = [866]
    return interesting_Data

if __name__ == '__main__':

     # obtain the benchmark suite    
    config_of_data = {      'Jad':{'data_ids':get_jad_data},
                            'OpenML': {'data_ids':get_openml_data}      }

    #opt_list = [('HEBO_RF_Local',Bayesian_Optimization)]
    opt_list = ['RF_Local']
    #'Multi_RF_Local','
    #'Random_Search'
    for repo in ['OpenML','Jad']:
        #XGBoost Benchmark    
        xgb_bench_config =  {
            'n_init' : 10,
            'max_evals' : 500,
            'n_datasets' : 1000,
            'data_ids' :  config_of_data[repo]['data_ids'](),
            'n_seeds' : [1,2,3],
            'type_of_bench': 'Multi_Fold_Single_Space_Results',
            'bench_name' :'XGB',
            'bench_class' : MultiFold_XGBoostBenchmark,
            'data_repo' : repo
        }
        run_benchmark_total(opt_list,xgb_bench_config)

    



    