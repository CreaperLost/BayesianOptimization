#A simple fetcher of OpenML data descriptions.
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
from get_pass import get_pass


def get_jad_data_list():
    min_samples = 10
    max_samples = 5000000
    min_features = 1
    max_features = 1000000
    analysis_type = 'REGRESSION'

    # # # Select dataset # # #
    # Initialise client
    ip, email, password =  get_pass()
    print(ip)
    print(email)
    print(password)
    jad = ApiClient(ip, email, password)

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


get_jad_data_list()