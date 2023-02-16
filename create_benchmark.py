import openml
import openml
import pandas as pd
from benchmarks.RandomForestBenchmark import RandomForestBenchmark
from pathlib import Path
from BayesianOptimizers.SMAC.smac_base import Bayesian_Optimization
from BayesianOptimizers.SMAC.Random_Search import Random_Search
import os
import sys
sys.path.insert(0, '..')
from benchmarks.XGBoostBenchmark import XGBoostBenchmark
from global_utilities.global_util import csv_postfix,directory_notation,file_name_connector,break_config_into_pieces_for_plots,parse_directory
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

    curr_dataset = 0

    #Add the type of Dataset repo before.
    score_directory = parse_directory([score_directory,data_repo])
    surrogate_time_directory = parse_directory([surrogate_time_directory,data_repo])
    objective_time_directory = parse_directory([objective_time_directory,data_repo])
    acquisition_time_directory = parse_directory([acquisition_time_directory,data_repo])

    for task_id in data_ids:
        
        #These Folders will have seed results in them.
        score_per_seed_directory = parse_directory([score_directory,'Dataset' +str(task_id)])
        surrogate_time_per_seed_directory  = parse_directory([surrogate_time_directory, 'Dataset' +str(task_id)])
        objective_time_per_seed_directory  = parse_directory([objective_time_directory, 'Dataset' +str(task_id)])
        acquisition_time_per_seed_directory  = parse_directory([acquisition_time_directory, 'Dataset' +str(task_id)])

        for seed in n_seeds:

            score_per_optimizer_directory = parse_directory([score_per_seed_directory,'Seed' + str(seed) ])
            surrogate_time_per_optimizer_directory = parse_directory([surrogate_time_per_seed_directory,'Seed' + str(seed) ])
            objective_time_per_optimizer_directory = parse_directory([objective_time_per_seed_directory,'Seed' + str(seed) ])
            acquisition_time_per_optimizer_directory = parse_directory([acquisition_time_per_seed_directory,'Seed' + str(seed) ])


            for opt in optimizers_list: # tuple opt[0] == name of opt , opt[1] == base_class of opt.
                benchmark_ = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo)
                configspace = benchmark_.get_configuration_space()
                

                print('Currently running ' + opt[0] + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )


                if opt[0] == 'RS':
                    Optimization = opt[1](f=benchmark_.objective_function,configuration_space= configspace,n_init = n_init,max_evals= max_evals,random_seed=seed)
                else:
                    Optimization = opt[1](f=benchmark_.objective_function,model=opt[0],lb= None, ub =None , configuration_space= configspace ,\
                    initial_design=None,n_init = n_init,max_evals= max_evals, batch_size=1 ,verbose=True,random_seed=seed)
                
                
                best_score = Optimization.run()

                #Get the evaluatinons.
                y_evaluations = Optimization.fX

                #Change this.
                surrogate_time_evaluations = Optimization.surrogate_time
                objective_time_evaluations= Optimization.objective_time
                acquisition_time_evaluations = Optimization.acquisition_time
                

                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=False)
                        Path(surrogate_time_per_optimizer_directory).mkdir(parents=True, exist_ok=False)
                        Path(objective_time_per_optimizer_directory).mkdir(parents=True, exist_ok=False)
                        Path(acquisition_time_per_optimizer_directory).mkdir(parents=True, exist_ok=False)
                    except FileExistsError:
                        print("Folder is already there")
                        """"""
                    else:
                        print("Folder is created there")
                        """"""
                        #print("Folder was created")
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt[0]+csv_postfix ]))
                    pd.DataFrame(surrogate_time_evaluations).to_csv( parse_directory([ surrogate_time_per_optimizer_directory, opt[0]+csv_postfix ]))
                    pd.DataFrame(objective_time_evaluations).to_csv( parse_directory([ objective_time_per_optimizer_directory, opt[0]+csv_postfix ]))
                    pd.DataFrame(acquisition_time_evaluations).to_csv( parse_directory([ acquisition_time_per_optimizer_directory, opt[0]+csv_postfix ]))
        # Just in case we want less.
        curr_dataset+=1
        if curr_dataset >= n_datasets:
            break 

def get_openml_data():
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    return benchmark_suite.tasks

def get_jad_data():
    return [1114,865,852,851,850,842,839,847,855,857,861,863,969,1048]

if __name__ == '__main__':

     # obtain the benchmark suite    
    config_of_data = {
        'Jad':{
            'data_ids':get_jad_data
        },
        'OpenML': {
            'data_ids':get_openml_data
        }
    }

    type_of_bench = 'Single_Space_Results'
    n_datasets = 20
    n_init = 20
    max_evals = 100
    repo = 'Jad'  #Jad
    seeds = [1,2,3] # ,2,3,4,5 
    """
    XGBoost Benchmark
    
    """
    xgb_bench_config =  {
        'n_init' : n_init,
        'max_evals' : max_evals,
        'n_datasets' : n_datasets,
        'data_ids' : config_of_data[repo]['data_ids'](),
        'n_seeds' : seeds,
        'type_of_bench': type_of_bench,
        'bench_name' :'XGB',
        'bench_class' : XGBoostBenchmark,
        'data_repo' : repo
    }
    #('RS',Random_Search),('RF',Bayesian_Optimization),('GP',Bayesian_Optimization)
    #('HEBO_RF',Bayesian_Optimization), ('HEBO_GP',Bayesian_Optimization)
    # ('RS',Random_Search),('RF',Bayesian_Optimization),('GP',Bayesian_Optimization),('GP',Bayesian_Optimization),('RS',Random_Search),
    run_benchmark_total([('HEBO_GP',Bayesian_Optimization),('HEBO_RF',Bayesian_Optimization),('GP',Bayesian_Optimization),('RF',Bayesian_Optimization),('RS',Random_Search)],xgb_bench_config)



    """
    
    RF Benchmark


    """

    """rf_bench_config =  {
        'n_init' : n_init,
        'max_evals' : max_evals,
        'n_datasets' : n_datasets,
        'data_ids' : config_of_data[repo]['data_ids'](),
        'n_seeds' : seeds,
        'type_of_bench':type_of_bench,
        'bench_name' : 'RF',
        'bench_class' : RandomForestBenchmark,
        'data_repo' : repo
    }

    run_benchmark_total([('GP',Bayesian_Optimization),('RF',Bayesian_Optimization)],rf_bench_config)"""