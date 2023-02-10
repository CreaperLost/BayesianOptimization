import openml
import openml
import pandas as pd
from benchmarks.RandomForestBenchmark import RandomForestBenchmarkBB
import argparse
from time import time
import logging
from pathlib import Path
from time import time
import numpy as np
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.scenario.scenario import Scenario
from smac.callbacks import IncorporateRunResultCallback
from BayesianOptimizers.SMAC.smac_base import BO_RF
from hpobench.util.example_utils import set_env_variables_to_use_only_one_core
import os
from BayesianOptimizers.SMAC.smac_base import BO_RF

"""logger = logging.getLogger("minicomp")
logging.basicConfig(level=logging.INFO)
set_env_variables_to_use_only_one_core()"""

benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite


def random_configuration(max_evals,benchmark):
    res_list = []
    time_list = []
    start_time = time()
    configurations = benchmark.get_configuration_space().sample_configuration(max_evals) 
    init_overhead = time() - start_time
    for config in configurations:
        start_time = time()
        result_dict = benchmark.objective_function(config.get_dictionary())
        res_list.append(result_dict['function_value'])
        end_time = time() - start_time
        if init_overhead != 0:
            end_time = end_time+init_overhead
            init_overhead = 0

        time_list.append(end_time)

    return res_list,time_list




def run_experiment_local_Smac():
    task_ids = benchmark_suite.tasks
    current_dataset = 0
    max_data = 100
    res_list = []
    res_list_time =[]

    n_init = 20
    max_evals = 100
    #seed  =1

    opt_list = ['GP']

    parent_path = os.getcwd() + '\Results'
    parent_path_time = os.getcwd() + '\Results_Time'
    for task_id in task_ids:
        print(f'# ################### TASK of {len(task_ids)}: Task-Id: {task_id} ################### #')
        if task_id == 167204:
                continue  # due to memory limits 
        current_dataset+=1
        dataset_path  = parent_path + '\Dataset'  +str(task_id) 
        dataset_path_time =parent_path_time + '\Dataset'  +str(task_id) 
        #Make the dataset path.
        if os.path.exists(dataset_path) == False:
            os.mkdir(dataset_path)
        if os.path.exists(dataset_path_time) == False:
            os.mkdir(dataset_path_time)
        for seed in [1,2,3,4,5]:

            b = RandomForestBenchmarkBB(task_id=task_id,rng=seed)
            cs = b.get_configuration_space()

            seed_path = dataset_path+'\Seed' + str(seed) 
            seed_path_time = dataset_path_time +'\Seed' + str(seed) 
            if os.path.exists(seed_path) == False:
                os.mkdir(seed_path)
            if os.path.exists(seed_path_time) == False:
                os.mkdir(seed_path_time)

            for opt_type in opt_list:
                # BO Opt.
                start = time()
                if opt_type == 'RF':
                    BO=BO_RF(f = b.objective_function , model='RF' ,lb= None, ub =None , configuration_space= cs ,\
                    initial_design=None,n_init = n_init,max_evals= max_evals, batch_size=1 ,verbose=True,random_seed=seed)
                    best_score=BO.run()
                    y_evaluations = BO.fX
                    print (y_evaluations)
                if opt_type == 'GP':
                    BO=BO_RF(f = b.objective_function ,model='GP' ,lb= None, ub =None , configuration_space= cs ,\
                    initial_design=None,n_init = n_init,max_evals= max_evals, batch_size=1 ,verbose=True,random_seed=seed)
                    best_score=BO.run()
                    y_evaluations = BO.fX
                    print (y_evaluations)
                elif opt_type == 'RS':
                    y_evaluations,time_eval = random_configuration(max_evals,b)
                    best_score = np.min(np.array(y_evaluations))
                end_time = time()-start
                print(opt_type + '__\n')
                print(f'Best Score {best_score}')
                print(f'Done, took totally {end_time:.2f}')
                
                
                pd.DataFrame(y_evaluations).to_csv( seed_path + '/' + opt_type + '.csv')
                #pd.DataFrame(time_eval).to_csv( seed_path_time + '/' + opt_type + '.csv')
                
        if current_dataset == max_data:
            break
    

if __name__ == '__main__':
    #run_experiment_random_search(on_travis=args.on_travis)
    run_experiment_local_Smac()
