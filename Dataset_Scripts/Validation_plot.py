from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
import sys
from dataset_utils import filter_datasets
import itertools
sys.path.insert(0, '..')
from global_utilities.global_util import csv_postfix,directory_notation,file_name_connector,break_config_into_pieces_for_plots,parse_directory
from pathlib import Path
import os
import csv
import scipy.stats as stats

#Casually returns the dataset name given a string
# e.g. Dataset11 --> Kr vs Kp
def get_dataset_name_OpenML(dataset,path):
    task_id = dataset.split('Dataset')[1] 
    task = openml.tasks.get_task(task_id, download_data=False)
    dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    return dataset_info.name

def get_dataset_name_Jad(dataset,path):
    data_id = dataset.split('Dataset')[1] 
    if not Path(path).exists():
        #fetch_jad_list()
        pass
    jad_datasets = pd.read_csv(path)    
    name = filter_datasets(jad_datasets,[int(data_id)],'Jad')['name']
    return name.values[0]
 
def get_dataset_name(dataset_id, config):
    result_space, classifier ,results_type ,optimizer_type, number_of_seeds, data_repo  = break_config_into_pieces_for_plots(config)
    #Get us the main file
    main_directory =  getcwd()

    #Briskomaste sto fakelo me kathe dataset kai ta results tou.

    if data_repo == 'Jad':
        path = parse_directory([main_directory,'Jad_Full_List.csv'])
        return get_dataset_name_Jad(dataset_id,path)
    
    #Get the directory.
    wanted_directory_attributes = [main_directory,result_space,classifier,results_type,data_repo,'dataset_characteristics.csv']
    path= parse_directory(wanted_directory_attributes)
    return get_dataset_name_OpenML(dataset_id,path)

def get_dataset_name_byrepo(dataset_id, data_repo):
    #Briskomaste sto fakelo me kathe dataset kai ta results tou.
    if data_repo == 'Jad':
        path = parse_directory([getcwd(),'Jad_Full_List.csv'])
        return get_dataset_name_Jad(dataset_id,path)
    
    return get_dataset_name_OpenML(dataset_id,None)

def save_figure(data_repo, dataset, time_plot_bool, clf_name):
    main_directory =  getcwd().replace(directory_notation+'Dataset_Scripts','')
    
    if data_repo == 'OverAllDatasets':
        path_to_figure = os.path.join(main_directory,'Figures','OverAllDatasets')
    else:
        path_to_figure = os.path.join(main_directory,'Figures',data_repo,dataset)


    if time_plot_bool == True:
        extra  = 'TimePlot'
    else:
        extra = 'SingleEvalPlot' 

    results_directory = os.path.join(path_to_figure,extra)
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    else:
        pass
    plt.savefig(parse_directory([results_directory,clf_name+'.png']),bbox_inches='tight')

def time_plot_for_opt(time,metric,opt):

    if 'SMAC' in opt:
        x = time['Time'].values
        y = time['Score'].values
    else:
        x = np.cumsum(time.values)
        y = np.minimum.accumulate(metric.values)
        
    return x,y


def config_plot_for_opt(metric,opt):

    x =  [i for i in range(metric.shape[0])]
    y = metric.values
    
    return x,y

def find_index(metric):
    new_array = np.zeros(metric.shape,dtype=np.int64)

    min_index = 0  # Initialize the minimum index
    for i in range(len(metric)):
        if metric[i] < metric[min_index]:
            min_index = i
        new_array[i] = min_index
    return new_array

def cumulative_min(metric):
    return np.minimum.accumulate(metric)


def value_per_index(test_score,index):
    new_array = np.zeros(test_score.shape)
    
    for i,value in enumerate(index): 
        new_array[i] = test_score.values[value[0]]

    return new_array

colors = ['red','blue','green','black','purple','orange','grey','cyan','yellow']
seeds = [1]
#How many initial configurations we have run.
interval = 50
result_space = 'Holdout_Experiments'
optimizers = ['Multi_RF_Local','Progressive_BO'] # 'Multi_RF_Local','Progressive_BO'

space_type = 'GROUP'

opt_colors = dict()
clr_pos = 0
for opt in optimizers:
    opt_colors.update({opt:colors[clr_pos]})
    clr_pos+=1


metric_per_data = dict()
metric_test_per_data =  dict()
for opt in optimizers:
    metric_per_data[opt] = []
    metric_test_per_data[opt] = []


for data_repo in ['Jad','OpenML']:
    path_str = os.path.join(os.pardir,result_space,space_type,'Metric',data_repo)
    if os.path.exists(path_str) == False:
        continue
    for time_bool_flag in [False]: #'True'
        for dataset in os.listdir(path_str):
            dataset_name = get_dataset_name_byrepo(dataset,data_repo)
            for seed in seeds:
                for opt in optimizers:
                    
                    metric=pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Metric',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])
                    metric.columns = ['Score']
                
                    test_metric=pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Metric_Test',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])
                    metric.columns = ['Score']

                    
                    
                    x,y = config_plot_for_opt(metric,opt)
                    index_of_min = find_index(y)


                    y  = cumulative_min(y)

                    y_test = value_per_index(test_metric,index_of_min)
                    
                    metric_per_data[opt].append(y.flatten())
                    metric_test_per_data[opt].append(y_test.flatten())

                    plt.plot(x,y_test,color = opt_colors[opt],linestyle='--',label=opt+' Test')
                    plt.plot(x,y,color=opt_colors[opt],linestyle ='solid',label=opt+ ' Train')
            plt.xlim([0,1050])
            plt.xlabel('Number of objective evals.')
            plt.grid(True, which='major')
            plt.title('Effectiveness of BO methods for dataset ' + dataset_name)
            plt.ylabel('1-AUC score')
            plt.legend()
            save_figure(data_repo,dataset_name,time_bool_flag,'Validation')
            plt.clf()


avg_train = dict()
avg_test =  dict()
for opt in optimizers:
    avg_train[opt] = list(pd.DataFrame(metric_per_data[opt]).transpose().mean(axis=1))
    avg_test[opt] = list(pd.DataFrame(metric_test_per_data[opt]).transpose().mean(axis=1))


    x =  [i for i in range(len(avg_train[opt]))]
    plt.plot(x,avg_test[opt],color = opt_colors[opt],linestyle='--',label=opt +' Test')
    plt.plot(x,avg_train[opt],color=opt_colors[opt],linestyle ='solid',label=opt+ ' Train')

plt.xlim([0,1050])
plt.xlabel('Number of objective evals.')
plt.grid(True, which='major')
plt.title('Effectiveness of BO methods for dataset')
plt.ylabel('Average (1-AUC) score')
plt.legend()
save_figure('Overall',dataset_name,time_bool_flag,'Validation')
plt.clf()