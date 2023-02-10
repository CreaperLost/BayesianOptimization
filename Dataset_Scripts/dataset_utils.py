from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
import os 
from pathlib import Path
import sys
sys.path.insert(0, '..')
from global_utilities.global_util import directory_notation,file_name_connector,break_config_into_pieces,parse_directory

#A function that opens the desired configuration,
#and after a API call returns the dataset ids associated with an analysis.
def get_dataset_ids(config = {}):
    assert config != {}

    #Get the configurations partials.
    result_space, classifier ,results_type   = break_config_into_pieces(config)

    #Get the main Bayesian Optimization directory.
    #Keep this a global.
    main_directory =  getcwd().replace('\\Dataset_Scripts','')

    wanted_directory_attributes = [main_directory,result_space,classifier,results_type]
    results_directory= parse_directory(wanted_directory_attributes)

    #Get the task ids from the datasets we run till now.
    Dataset_files = [f for f in listdir(results_directory) if isdir(join(results_directory, f))]
    task_ids= [dataset.split('Dataset')[1] for dataset in  Dataset_files]
    task_info = openml.tasks.get_tasks(task_ids, download_data=False)
    data_ids = [i.dataset_id for i in task_info]
    return data_ids

#A simple fetcher of OpenML data descriptions.
def get_open_ml_data_list():
    #Get the list of datasets from openml
    datalist = pd.DataFrame.from_dict(openml.datasets.list_datasets(), orient="index")
    return datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

#Filtering out the UN-WANTED ids from the OpenMl CSV.
def filter_datasets(datalist,data_ids):
    #Return--Save only the datasets we already run.. :)
    return datalist[datalist['did'].isin(data_ids)]
    
def save_info(data_info,config):
    
    #Get the configurations partials.
    result_space, classifier ,results_type   = break_config_into_pieces(config)
    
    #Get the main Bayesian Optimization directory.
    #Keep this a global.
    main_directory =  getcwd()

    wanted_directory_attributes = [main_directory,result_space,classifier,results_type]
    results_directory = parse_directory(wanted_directory_attributes)
    
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
            
    data_info.to_csv(results_directory + directory_notation + 'dataset_characteristics.csv',index=False)
