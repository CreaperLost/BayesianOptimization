from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
from dataset_utils import get_dataset_ids,get_open_ml_data_list,filter_datasets,save_info


config_list = [dict( results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

for config in config_list:
    data_id = get_dataset_ids(config)
    open_ml_data  =get_open_ml_data_list()
    remaining_data =filter_datasets(datalist=open_ml_data,data_ids=data_id) 
    save_info(remaining_data,config)