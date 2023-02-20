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
from global_utilities.global_util import directory_notation,file_name_connector,break_config_into_pieces_for_plots,parse_directory
from jadbio_api.api_client import ApiClient


#A function that opens the desired configuration,
#and after a API call returns the dataset ids associated with an analysis.
def get_dataset_ids(config = {}):
    assert config != {}

    #Get the configurations partials.
    result_space, classifier ,results_type,optimizer_type,seeds,data_repo   = break_config_into_pieces_for_plots(config)

    #Get the main Bayesian Optimization directory.
    #Keep this a global.
    main_directory =  getcwd().replace('\\Dataset_Scripts','')

    wanted_directory_attributes = [main_directory,result_space,classifier,results_type,data_repo]
    results_directory= parse_directory(wanted_directory_attributes)

    #Get the task ids from the datasets we run till now.
    Dataset_files = [f for f in listdir(results_directory) if isdir(join(results_directory, f))]
    print(Dataset_files)
    task_ids= [dataset.split('Dataset')[1] for dataset in  Dataset_files]

    #Sobaro check here. OpenML == taskid, eno sto Jad == Data_id
    if data_repo == 'OpenML':
        task_info = openml.tasks.get_tasks(task_ids, download_data=False)
        data_ids = [i.dataset_id for i in task_info]
    else:
        data_ids =  [int(task_id) for task_id in task_ids]
    return data_ids

def get_data_list(data_repo):
    if data_repo == 'Jad':
        return get_jad_data_list()
    return get_open_ml_data_list()

#A simple fetcher of OpenML data descriptions.
def get_open_ml_data_list():
    #Get the list of datasets from openml
    datalist = pd.DataFrame.from_dict(openml.datasets.list_datasets(), orient="index")
    return datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

#A simple fetcher of OpenML data descriptions.
def get_jad_data_list():
    min_samples = 10
    max_samples = 5000000
    min_features = 1
    max_features = 1000000
    analysis_type = 'CLASSIFICATION'

    # # # Select dataset # # #
    # Initialise client
    jad = ApiClient('https://exp.jadbio.com:4443', 'pkatsogr@gmail.com', '22222222')

    project = jad.project.find_project('jad_research')
    # Load dataset list
    dataset_list = jad.project.find_project_datasets(project)
    if analysis_type == 'CLASSIFICATION':
        dataset_list = [d for d in dataset_list if jad.project.find_project_dataset_feature(d['id'], 'target')['type'] == 'Categorical']
    else:
        dataset_list = [d for d in dataset_list if jad.project.find_project_dataset_feature(d['id'], 'target')['type'] == 'Numerical']
    dataset_list = [dataset_list[d] for d in range(len(dataset_list))
                if (dataset_list[d]['samples'] >= min_samples) & (dataset_list[d]['samples'] <= max_samples) &
                (dataset_list[d]['features'] >= min_features) & (dataset_list[d]['features'] <= max_features)]
    df = pd.DataFrame(dataset_list,columns=list(dataset_list[0].keys()))
    return df


def filter_datasets(datalist,data_ids,repo):
    if repo == 'Jad':
        return filter_Jad_datasets(datalist,data_ids)
    return filter_openml_datasets(datalist,data_ids)

#Filtering out the UN-WANTED ids from the OpenMl CSV.
def filter_openml_datasets(datalist,data_ids):
    #Return--Save only the datasets we already run.. :)
    
    return datalist[datalist['did'].isin(data_ids)]
    

def filter_Jad_datasets(datalist,data_ids):
    
    return datalist[datalist['id'].isin(data_ids)]

def save_info(data_info,config):
    
    #Get the configurations partials.
    result_space, classifier ,results_type,optimizer_type,seeds,data_repo   = break_config_into_pieces_for_plots(config)
    
    #Get the main Bayesian Optimization directory.
    #Keep this a global.
    main_directory =  getcwd()

    #Sozeis ta data sto repo to katalilo.
    wanted_directory_attributes = [main_directory,result_space,classifier,results_type,data_repo]
    results_directory = parse_directory(wanted_directory_attributes)
    
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
            
    data_info.to_csv(results_directory + directory_notation + 'dataset_characteristics.csv',index=False)
