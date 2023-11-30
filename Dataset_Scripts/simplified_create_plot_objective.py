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
    y = np.minimum.accumulate(metric.values)
    
    return x,y
    

def get_Jad_avg_score(dataset_name):
    parent_dir = os.path.join(os.getcwd(), os.pardir)
    Res_File = pd.read_csv(os.path.join(parent_dir,'JAD_Results_AUC.csv'),index_col=0).set_index('dataset')
    if dataset_name not in Res_File.index:
        return None
    return Res_File.loc[dataset_name].values[0]

colors = ['red','blue','green','black','purple','orange','grey','cyan','yellow']
seeds = [1]
#How many initial configurations we have run.
interval = 50
result_space = 'Main_Multi_Fold_Group_Space_Results'
space_type = 'GROUP'

#optimizers = ['Mango','Multi_RF_Local','Random_Search','SMAC','Progressive_BO','RF_Local','Optuna','Mango2'] # 'Multi_RF_Local','Greedy_SM',
#optimizers = ['RF_Local','Hyperband','Progressive_BO','SMAC_Instance','Greedy_SM']
              #'Random_Search','Mango2','Optuna','SMAC','HyperOpt','
#optimizers = ['RF_Local', 'RF_Local_No_STD', 'RF_Local_extensive', ]

optimizers = ['SMAC', 'SMAC_Instance']


# 'RF_Local_No_STD', 'RF_Local_extensive', 
opt_colors = dict()
clr_pos = 0
for opt in optimizers:
    opt_colors.update({opt:colors[clr_pos]})
    clr_pos+=1

"""for data_repo in ['Jad','OpenML']:
    path_str = os.path.join(os.pardir,result_space,space_type,'Metric',data_repo)
    if os.path.exists(path_str) == False:
        continue
    for time_bool_flag in [False]: #'True'
        for dataset in os.listdir(path_str):
            dataset_name = get_dataset_name_byrepo(dataset,data_repo)
            for seed in seeds:
                for opt in optimizers:
                    if opt == 'Jad':
                        jad_score =  get_Jad_avg_score(dataset_name)
                        if jad_score != None: plt.axhline(y= 1-jad_score,label=opt)
                        continue
                    
                    try:
                        metric=pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Metric',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])
                        metric.columns = ['Score']
                    except:
                        continue
                    if time_bool_flag:
                        time = pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Total_Time',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])  
                        if 'SMAC' not in opt:
                            time.columns = ['Time']
                        else:
                            time.columns = ['Time','Score']

                        
                        plt.xlabel('Average time in seconds')
                        x,y = time_plot_for_opt(time,metric,opt)
                        
                    else:
                        plt.xlim([0,1050])
                        plt.xlabel('Number of objective evals.')
                        x,y = config_plot_for_opt(metric,opt)
                    if opt == 'Greedy_SM':
                        plt.plot([i for i in range(800,1050)],y,opt_colors[opt],label=opt)
                    else:
                        print(opt)
                        plt.plot(x,y,opt_colors[opt],label=opt)

            plt.grid(True, which='major')
            plt.title('Effectiveness of BO methods for dataset ' + dataset_name)
            plt.ylabel('1-AUC score')
            plt.legend()
            save_figure(data_repo,dataset_name,time_bool_flag,'Group')
            plt.clf()"""
                  

# Store the results per optimizer.
y_per_opt_for_config = {}
x_per_opt_for_config = {}
# Store the time results per optimizer.
y_per_opt_for_time = {}
x_per_opt_for_time = {}
for opt in optimizers:
    y_per_opt_for_config[opt] = []
    x_per_opt_for_config[opt] = []
    y_per_opt_for_time[opt] = []
    x_per_opt_for_time[opt] = []

dataset_list = []

for data_repo in ['Jad','OpenML']:
    # If the repository doesn't exist then move on.
    print(data_repo)
    path_str = os.path.join(os.pardir,result_space,space_type,'Metric',data_repo)
    if os.path.exists(path_str) == False: continue
    for dataset in os.listdir(path_str):
        
        dataset_name = get_dataset_name_byrepo(dataset,data_repo)
        dataset_list.append(dataset_name +'_' +  str(dataset))
        for seed in seeds:
            for opt in optimizers:
                if opt == 'Jad':
                    jad_score =  get_Jad_avg_score(dataset_name)
                    if jad_score != None: 
                        y_per_opt_for_config[opt].append(1 - jad_score)
                        y_per_opt_for_time[opt].append(1- jad_score)
                    continue
                        
                try:
                    metric=pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Metric',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])
                    metric.columns = ['Score']
                except:
                    continue
                """time = pd.read_csv(os.path.join(os.pardir,result_space,space_type,'Total_Time',data_repo,dataset,'Seed'+str(seed),opt,opt+'.csv'),index_col=['Unnamed: 0'])  
                if 'SMAC' not in opt:
                    time.columns = ['Time']
                else:
                    time.columns = ['Time','Score']"""
                        
                #x_time,y_time = time_plot_for_opt(time,metric,opt)
                x,y = config_plot_for_opt(metric,opt)
                
                y_per_opt_for_config[opt].append(y)
                x_per_opt_for_config[opt].append(x)
                """y_per_opt_for_time[opt].append(y_time)
                x_per_opt_for_time[opt].append(x_time)"""


        # Define the desired confidence level (e.g., 95% confidence interval)

def get_confidence_interval(row):
    mean = row.mean()
    confidence_level = 0.95
    std_error = stats.sem(row)
    interval = stats.t.interval(confidence_level, len(row)-1, loc=mean, scale=std_error)
    return interval 

def compute_row_mean_and_std(dictionary_entry,iter,opt):
    # Create an empty DataFrame
    df = pd.DataFrame()
    # Iterate through the array_list and append each array as a column
    
    for i, arr in enumerate(dictionary_entry):
        
        my_array  = arr.flatten()
        if opt == 'Progressive_BO':
            tmp_arr = []
            # [25,75,150,250,500]
            for idx,val in enumerate(my_array):
                if idx  < 75:
                    tmp_arr.append(val)
                elif idx == 75:
                    for _ in range(idx):
                        tmp_arr.append(val)

                elif idx < 150:
                    for _ in range(2):
                        tmp_arr.append(val)
                elif idx == 150:
                    for _ in range(idx):
                        tmp_arr.append(val)

                elif idx < 300:
                    for _ in range(3):
                        tmp_arr.append(val)
                elif idx == 300:
                    for _ in range(idx):
                        tmp_arr.append(val)

                elif idx < 550:
                    for _ in range(4):
                        tmp_arr.append(val)
                elif idx == 550:
                    for _ in range(idx):
                        tmp_arr.append(val)

                elif idx < 1050:
                    for _ in range(5):
                        tmp_arr.append(val)
                elif idx == 1050:
                    for _ in range(idx):
                        tmp_arr.append(val)
            my_array = tmp_arr        
        elif opt == 'SMAC_Instance':
            pass
        else:
            my_array = [value for value in my_array for _ in range(5)]
        print('Number of iterations... ',len(my_array),opt)
        while len(my_array) < iter:
            my_array = np.append(my_array, my_array[-1])
        df[f'Column {i+1}'] = my_array

    

    # Compute the mean of each row
    row_means = df.mean(axis=1)
    # Apply the function across each row to calculate the confidence interval
    confidence_intervals = df.apply(get_confidence_interval, axis=1, result_type='expand')
    # Rename the columns
    confidence_intervals.columns = ['CI Lower', 'CI Upper']
    # Display the row means
    result = pd.concat([row_means, confidence_intervals], axis=1)
    result.columns = ['Mean','Low','Upper']
    return df,result



def greedy_score(mean_score):
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Iterate through the array_list and append each array as a column
    
    for i, arr in enumerate(mean_score):
        my_array  = arr.flatten()
        tmp_arr = []
        # This is not entirely correct. The previous configurations have already run.
        for idx,val in enumerate(my_array):
            if idx < 10:
                tmp_arr.append(val)
            else:
                for _ in range(5):
                    tmp_arr.append(val)
        my_array = tmp_arr
        print('Number of iterations... Greedy',len(my_array))
        df[f'Column {i+1}'] = my_array

    final_scores = [0 for i in range(2850)] + list(df.mean(axis=1))
    
    return final_scores


time_bool_flag = False
for opt in optimizers:
    print(f'Current Optimizer {opt}')
    x=[i for i in range(0,5250)]
    if opt =='Jad':
        continue
    else:
        if opt == 'Greedy_SM':
            plt.plot([i for i in range(2850,3960)],greedy_score(y_per_opt_for_config[opt])[2850:3960],opt_colors[opt],label=opt)
        else:
            df,result = compute_row_mean_and_std(y_per_opt_for_config[opt],5250,opt)
            plt.plot(x,result['Mean'],opt_colors[opt],label=opt)

plt.ylim([0.08,0.1])
plt.xlim([0,5250])
plt.xlabel('Number of objective evals.')
plt.grid(True, which='major')
plt.title('Effectiveness of BO methods for all datasets')
plt.ylabel('Average 1-AUC')
plt.legend()
save_figure('OverAllDatasets',dataset_name,time_bool_flag,'Group')
plt.clf()


"""
def compute_percentile(numpy_time_measures):
    positions = [0]
    for i in range(1, 10):
        value = i / 10.0
        positions.append(int(len(numpy_time_measures) * value))
    positions.append(len(numpy_time_measures) - 1)


    results = []
    for i in positions:
        results.append(numpy_time_measures[i])
    
    return np.array(results)
    # Calculate positions
    start_position = 0
    position_25 = int(len(numpy_time_measures) * 0.25)
    position_50 = int(len(numpy_time_measures) * 0.5)
    position_75 = int(len(numpy_time_measures) * 0.75)
    final_position = len(numpy_time_measures) - 1
    # Compute percentiles
    start_percentile = numpy_time_measures[start_position]
    percentile_25 = numpy_time_measures[position_25]
    percentile_50 = numpy_time_measures[position_50]
    percentile_75 = numpy_time_measures[position_75]
    final_percentile = numpy_time_measures[final_position]
    return np.array([start_percentile,percentile_25,percentile_50,percentile_75,final_percentile])


def compute_mean_std_per_time(dictionary_entry):
    # Create an empty DataFrame
    df = pd.DataFrame()
    # Iterate through the array_list and append each array as a column
    for i, arr in enumerate(dictionary_entry):

        df[f'Column {i+1}'] = compute_percentile(arr.flatten())

    # Compute the mean of each row
    row_means = df.mean(axis=1)
    # Apply the function across each row to calculate the confidence interval
    confidence_intervals = df.apply(get_confidence_interval, axis=1, result_type='expand')
    # Rename the columns
    confidence_intervals.columns = ['CI Lower', 'CI Upper']
    # Display the row means
    result = pd.concat([row_means, confidence_intervals], axis=1)
    result.columns = ['Mean','Low','Upper']
    return result



def compute_avg_time(dictionary_entry):
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through the array_list and append each array as a column
    for i, arr in enumerate(dictionary_entry):
        #print(i,arr)
        df[f'Column {i+1}'] = compute_percentile(arr.flatten())
    # Compute the mean of each row
    row_means = list(df.mean(axis=1))
    return row_means


time_bool_flag = True
for opt in optimizers:
    print(f'Current Optimizer {opt}')
    result = compute_mean_std_per_time(y_per_opt_for_time[opt])
    x = compute_avg_time(x_per_opt_for_time[opt])
    if opt =='Jad':
        continue
    plt.plot(x,result['Mean'],opt_colors[opt],label=opt)
    #plt.fill_between(x, result['Low'], result['Upper'],color=opt_colors[opt], alpha=0.1)
    #print(result)
plt.ylim([0.065,0.1])
plt.xlabel('Time')
plt.grid(True, which='major')
plt.title('Effectiveness of BO methods for all datasets ' )
plt.ylabel('Average 1-AUC')
plt.legend()
save_figure('OverAllDatasets',dataset_name,time_bool_flag,'Group')
plt.clf()
"""