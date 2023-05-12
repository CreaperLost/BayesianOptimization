import openml
import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
from pathlib import Path
from BayesianOptimizers.Conditional_BayesianOptimization.Group_Random_Search import Group_Random_Search
from BayesianOptimizers.Conditional_BayesianOptimization.Group_SMAC_base import Group_Bayesian_Optimization
from BayesianOptimizers.Experimental.Pavlos_BO import Pavlos_BO

import os
import sys
sys.path.insert(0, '..')
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np




#Casually returns the dataset name given a string
# e.g. Dataset11 --> Kr vs Kp
def get_dataset_name_OpenML(dataset):
    task_id = dataset
    task = openml.tasks.get_task(task_id, download_data=False)
    dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    return dataset_info.name

def get_dataset_name_Jad(dataset):
    data_id = dataset
    path = os.getcwd() + '/Dataset_Scripts/Jad_Full_List.csv'
    jad_datasets = pd.read_csv(path)    
    name = jad_datasets[jad_datasets['id'].isin([int(data_id)])]['name'].values[0]
    return name


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

    #Add the type of Dataset repo before.
    score_directory = parse_directory([score_directory,data_repo])



    total_res = np.array(['name','avg config','avg score'])
    for task_id in data_ids:
        
        #These Folders will have seed results in them.
        score_per_seed_directory = parse_directory([score_directory,'Dataset' +str(task_id)])
        
        if data_repo == 'Jad':
            name = get_dataset_name_Jad(task_id)
        else:
            name = get_dataset_name_OpenML(task_id)

        average_score = 0
        average_configurations = 0
        for seed in n_seeds:

            score_per_optimizer_directory = parse_directory([score_per_seed_directory,'Seed' + str(seed) ])
            
            for opt in optimizers_list: 
                
                results = pd.read_csv(parse_directory([score_per_optimizer_directory, opt+csv_postfix]),skiprows=1,names=['score'])
                
                average_score += results.min().values[0]
                average_configurations += results.shape[0]


        average_configurations = int(average_configurations/len(n_seeds))
        average_score = np.round(average_score/len(n_seeds),6)

        total_res = np.vstack ( (total_res, np.array([name,average_configurations,average_score]) ) )
        total_df=pd.DataFrame(total_res)
        headers = total_df.iloc[0]
        total_df  = pd.DataFrame(total_df.values[1:], columns=headers)
        print(total_df)
                
                
                

def get_openml_data(speed = None):
    # 2074 needs 15 hours for 3 seeds per optimizer.
    assert speed !=None
    if speed == 'fast':
        return [14954,11,3918,3917,3021,43,167141,9952]
    return [2074,9976,9910,167125]
    
    

#
def get_jad_data(speed = None):
    assert speed !=None
    if speed == 'fast':
        return [842,851,850] # 839, 847,1114,
    #  on all seeds 
    return [843,883,866]
    

if __name__ == '__main__':
    config_of_data = { 'Jad':{'data_ids':get_jad_data},
                        'OpenML': {'data_ids':get_openml_data}      }
    opt_list = ['Pavlos'] # ,'Multi_RF_Local' ,'Random_Search','RF_Local',]
    for speed in ['fast']:
     # obtain the benchmark suite    
        for repo in ['Jad','OpenML']: #,'Jad' ,
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

    



    