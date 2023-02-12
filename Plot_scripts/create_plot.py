from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
import sys
sys.path.insert(0, '..')
from global_utilities.global_util import csv_postfix,directory_notation,file_name_connector,break_config_into_pieces_for_plots,parse_directory
from pathlib import Path


def get_results_per_optimizer(config={}):
    assert config!={}

    #Get the configurations partials.
    result_space, classifier ,results_type ,optimizer_type, number_of_seeds  = break_config_into_pieces_for_plots(config)

    #Get us the main file
    main_directory =  getcwd().replace('\\Plot_scripts','')

    #Get the directory.
    wanted_directory_attributes = [main_directory,result_space,classifier,results_type]
    # Briskomaste sto fakelo me kathe dataset kai ta results tou.
    results_directory= parse_directory(wanted_directory_attributes)

    #Get each dataset file. :D
    Dataset_files = [f for f in listdir(results_directory) if isdir(join(results_directory, f))]
    
    #Save the results of the optimizer per dataset.
    optimizer_results_per_dataset = dict()

    for dataset in  Dataset_files:
        #Get in the dataset directory.
        seed_directory = parse_directory([results_directory,dataset])

        #Get all the seed files
        seed_files = [f for f in listdir(seed_directory) if isdir(join(seed_directory, f))]

        #Check seed files == n_seeds
        assert len(seed_files) == number_of_seeds
        
        seed_results = []
        #Get the optimizer result per seed file.
        for seed_file in seed_files:
            #Not needed anymore.
            #optimizer_files = [f for f in listdir(optimizer_path) if isfile(join(optimizer_path, f))]

            #Get the Optimizer file per seed.
            optimizer_path = parse_directory([seed_directory,seed_file,optimizer_type]) + csv_postfix

            #Open the optimizer file.
            opt_results = pd.read_csv(optimizer_path,index_col=0)
            
            #Create a accumulation.
            accumulation = np.minimum.accumulate(opt_results)
            
            #Save the results of the current seed.
            seed_results.append(accumulation)

        #Save the list of seeds for the current dataset
        optimizer_results_per_dataset[dataset] = { 'result' : seed_results }
    
    return optimizer_results_per_dataset

#Computes the bounds for each evaluation point, upper and lower confidence. a is the best, b is the worst.
#Returns dataframe. 1 row is the best, 2nd row is the worst.
# df.iloc[0,:] , df.iloc[1,:] to access
# returns also the mean
def get_seeds_per_dataset(seeds):
    df_concat = pd.concat(seeds, axis = 1)
    a,b=stats.norm.interval(0.95, loc=df_concat.mean(axis=1), scale=df_concat.std(axis=1)/np.sqrt(len(seeds)))
    return pd.DataFrame(np.vstack((a,b))) , df_concat.mean(axis=1)



#Casually returns the dataset name given a string
# e.g. Dataset11 --> Kr vs Kp
def get_dataset_name(dataset):
    task_id = dataset.split('Dataset')[1] 
    task = openml.tasks.get_task(task_id, download_data=False)
    dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    return dataset_info.name

def create_plot_per_optimizer(optimizer_results_for_dataset):
    confidence, means = get_seeds_per_dataset(optimizer_results_for_dataset['result'])
    return confidence,means
    
    
# Gia kathe configuration experiment.
# Gia kathe classifier.
# Gia kathe dataset.
# Gia kathe seed.
# Gia kathe optimizer.


opt_colors= {
    'RS':'r',
    'RF':'b',
    'GP':'g',
}

config_list = [dict( seeds = 3 , optimizer_type = 'RF',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
               ,dict( seeds = 3 , optimizer_type = 'GP',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
               ,dict( seeds = 3 , optimizer_type = 'RS',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

config_results =  [get_results_per_optimizer(config) for config in config_list]
datasets_list_run = config_results[0].keys()



#Take the datasets--Dirty.
for dataset in datasets_list_run:
    title_name = get_dataset_name(dataset)
    fig, ax = plt.subplots()
    for config_result_idx in range(len(config_results)):
        opt = config_list[config_result_idx]['optimizer_type']
        
        confidence,means = create_plot_per_optimizer(config_results[config_result_idx][dataset])
        eval_range = means.shape[0]
        x = [i+1 for i in range(eval_range)]
        ax.plot(x,means,opt_colors[opt],label=opt)
        ax.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)

    x_ticks = [i for i in range(eval_range) if i%10==0]
    plt.xticks(x_ticks,x_ticks)
    plt.xlim([1,eval_range])
    plt.axvline(x = 20, color = 'black', linestyle = '--',label='Initial_Evaluations')
    plt.title(title_name + " /w classifier "  + config_list[0]['classifier'])
    plt.ylabel('Error Rate (1-Accuracy)')
    plt.legend()
    #plt.show()
    
    main_directory =  getcwd().replace('\\Plot_scripts','')
    wanted_directory_attributes = [main_directory,'Figures',dataset]
    results_directory = parse_directory(wanted_directory_attributes)
    
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    plt.savefig(results_directory+directory_notation+config_list[0]['classifier']+'.png')