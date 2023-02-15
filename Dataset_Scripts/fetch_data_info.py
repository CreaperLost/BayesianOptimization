from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
from dataset_utils import get_dataset_ids,get_data_list,filter_datasets,save_info

data_repo = 'OpenML'
n_seeds = 5

config_list = [dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RF',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'GP',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RS',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

for config in config_list:
    data_id = get_dataset_ids(config)
    print(data_id)
    data_list  = get_data_list(config['data_repo'])
    print(data_list)
    remaining_data =filter_datasets(datalist=data_list,data_ids=data_id,repo = config['data_repo']) 
    save_info(remaining_data,config) 