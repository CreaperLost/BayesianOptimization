from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
import sys
from dataset_utils import filter_datasets

sys.path.insert(0, '..')
from global_utilities.global_util import csv_postfix,directory_notation,file_name_connector,break_config_into_pieces_for_plots,parse_directory
from pathlib import Path

def get_results_per_optimizer(config={},accumulate='none'):
    assert config!={}
    assert accumulate == 'none' or accumulate =='min' or accumulate =='max'

    #Get the configurations partials.
    result_space, classifier ,results_type ,optimizer_type, number_of_seeds, data_repo  = break_config_into_pieces_for_plots(config)

    #Get us the main file
    main_directory =  getcwd().replace('\\Dataset_Scripts','')

    #Get the directory.
    wanted_directory_attributes = [main_directory,result_space,classifier,results_type,data_repo]


    # Briskomaste sto fakelo me kathe dataset kai ta results tou.
    results_directory= parse_directory(wanted_directory_attributes)

    #Get each dataset file. :D
    Dataset_files = [f for f in listdir(results_directory) if isdir(join(results_directory, f))]

    #Save the  results of the optimizer per dataset.
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
            if accumulate == 'min':
                accumulation = np.minimum.accumulate(opt_results)
            elif accumulate == 'max':
                accumulation = np.maximum.accumulate(opt_results)
            else:
                accumulation = opt_results
            
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


def get_dataset_name(dataset_id, config):
    result_space, classifier ,results_type ,optimizer_type, number_of_seeds, data_repo  = break_config_into_pieces_for_plots(config)
    #Get us the main file
    main_directory =  getcwd()

    #Get the directory.
    wanted_directory_attributes = [main_directory,result_space,classifier,results_type,data_repo,'dataset_characteristics.csv']

    # Briskomaste sto fakelo me kathe dataset kai ta results tou.
    path= parse_directory(wanted_directory_attributes)

    print(path)

    if data_repo == 'Jad':
        return get_dataset_name_Jad(dataset_id,path)
    return get_dataset_name_OpenML(dataset_id,path)


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
        print('oops')
    jad_datasets = pd.read_csv(path)    
    name = filter_datasets(jad_datasets,[int(data_id)],'Jad')['name']
    return name.values[0]


def create_plot_per_optimizer(optimizer_results_for_dataset):
    confidence, means = get_seeds_per_dataset(optimizer_results_for_dataset['result'])
    return confidence,means
    
    
# Gia kathe configuration experiment.
# Gia kathe classifier.
# Gia kathe dataset.
# Gia kathe seed.
# Gia kathe optimizer.

opt_colors= {
    'RS':'red',
    'RF':'blue',
    'GP':'green',
    'HEBO_RF':'black',
    'HEBO_GP':'purple'
}

data_repo = 'Jad'
n_seeds=  3
optimizers = ['RF','GP','RS','HEBO_RF','HEBO_GP']
metrics = ['Metric','Surrogate_Time','Objective_Time','Acquisition_Time']
time_plot = True
double_plot = True

general_config = {
    'classifier':'XGB',
    'result_space':'Single_Space_Results',
    'optimizers' : optimizers,
    'n_seeds' : n_seeds,
    'data_repo':data_repo,
    'double_plot':double_plot,
    'metrics': metrics,
    'time_plot':time_plot,
}
#How many initial configurations we have run.
interval = 50  



def plot_per_dataset(config):
    clf_name = config['classifier']
    result_space = config['Single_Space_Results']
    opt_list = config['optimizers']
    seeds= config['n_seeds']
    data_repo = config['data_repo']
    metric_list = config['metrics']
    #Boolean checkers for what plot type we want.
    double_plot_bool = config['double_plot']
    time_plot_bool = config['time_plot']

    #We need at least 1 metric and 1 optimizer.
    assert len(metric_list) > 0 and len(opt_list) > 0

    total_config_dictionary = dict()

    for metric in metric_list:
        #Per metric.
        list_per_metric = []
        for opt_name in opt_list:
            #Create the dictionary for the optimizer and the specified metric
            config_dict_per_metric=dict( data_repo = data_repo, seeds = seeds , optimizer_type = opt_name,results_type = metric, classifier = clf_name, result_space = result_space)
            list_per_metric.append(config_dict_per_metric)
        #Add the optimizer configurations for the specific metric.
        total_config_dictionary[metric] = list_per_metric
    
    configuration_results = dict()
    #Get the list of configurations per metric.
    for metric_config_name in list(total_config_dictionary.keys()):
        configuration_list_for_a_metric = total_config_dictionary[metric_config_name]
        #Save the results for the specific configuration and the specific metric.
        #Whether to accumulate the results or not.
        if metric_config_name == 'Metric':
            acc= 'min'
        elif metric_config_name == 'Total_Cost':
            acc= 'max'
        else:
            acc= 'none'
        configuration_results[metric_config_name] = [get_results_per_optimizer(config,acc) for config in configuration_list_for_a_metric]

    #Get the list of datasets run on the metrics.
    datasets_list_run = configuration_results[metric_list[0]][0].keys()

    #Take the datasets--Dirty.
    for dataset in datasets_list_run:
        #Inserting the first of many configurations give_us acess to many of the fields we want. -- Dirty.
        title_name = get_dataset_name(dataset,configuration_results[metric_list[0]][0])
        if double_plot_bool == True:
            fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        else: 
            fig, ax1 = plt.subplots()
        #for each configuration in the configurations do:
        #At this point we essentially parse throught the optimizers.
        for config_result_idx in range(len(configuration_results[metric_list[0]])):
            if time_plot_bool == True:

                #The configuration list of results for the specific metric
                config_list_score = configuration_results['Metric']
                config_list_time = configuration_results['Total_Cost']
                
                #Get the optimizers.
                opt_1 = config_list_score[config_result_idx]['optimizer_type']
                opt_2 = config_list_time[config_result_idx]['optimizer_type']

                #Just make sure the optimizers are exactly the same.
                assert opt_1 == opt_2

                #Get the confidence and means for the specific dataset.
                confidence,means = create_plot_per_optimizer(config_list_score[config_result_idx][dataset])
                #Get the confidence in time and the mean metric for the specific dataset
                time_confidence,time_means = create_plot_per_optimizer(config_list_time[config_result_idx][dataset])

                x = config_list_time[config_result_idx][dataset]
                ax1.plot(time_means,means,opt_colors[opt],label=opt)
                ax1.fill_between(time_means, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)

            else:
                for metric_measured in metric_list: 
                    
                    #The configuration list of results for the specific metric
                    config_list_per_metric = configuration_results[metric_measured]

                    #Get the optimizer
                    opt = config_list_per_metric[config_result_idx]['optimizer_type']

                    if metric_measured == 'Metric' :

                        #Create a plot for the specific metric.
                        confidence,means = create_plot_per_optimizer(config_list_per_metric[config_result_idx][dataset])

                        #Get the range for the x-axis
                        eval_range = means.shape[0]
                        x = [i+1 for i in range(eval_range)]

                        #Plot the mean and the confidence
                        ax1.plot(x,means,opt_colors[opt],label=opt)
                        ax1.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)

                    elif double_plot_bool == False and metric_measured != 'Metric':
                        continue 
                    elif double_plot_bool == True and metric_measured != 'Metric':
                        #Create a plot for the surrogate time metric.
                        confidence,means = create_plot_per_optimizer(config_list_per_metric[config_result_idx][dataset])

                        #Check the metric we measure and do the appropriate handling. :)
                        if metric_measured == 'Objective_Time' and config_result_idx == 0:
                            ax2.plot(x,means,'orange',label='Mean-Objective Time')
                        elif metric_measured == 'Acquisition_Time' and config_result_idx == 0:
                            ax2.plot(x,means,'yellow',label='Mean-Acquisition Time')
                        #Plot the mean and the confidence on axis 2.
                        elif metric_measured == 'Surrogate_Time':
                            ax2.plot(x,means,opt_colors[opt],label=opt)
                            ax2.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)
                
                #Initial configurations.    
        if time_plot_bool == False:
            x_ticks = [i for i in range(eval_range) if i%interval==0]
            plt.xticks(x_ticks,x_ticks)
            plt.xlim([1,eval_range])
            plt.axvline(x = interval, color = 'black', linestyle = '--',label='Initial_Evaluations')
            plt.xlabel('Number of Objective Evaluations')

            if double_plot_bool == True:
                ax2.set_ylabel('Time in seconds')

        
        # This happens for all the figures.
        ax1.set_ylabel('(1-AUC)')
        fig.suptitle(title_name + " /w classifier "  + clf_name)
        plt.legend(bbox_to_anchor=(1.05, 1),loc = 'upper left')
                

        main_directory =  getcwd().replace('\\Dataset_Scripts','')
        if time_plot_bool == True:
            wanted_directory_attributes = [main_directory,'Figures',dataset,'TimePlot']
        elif time_plot_bool == False:
            if double_plot_bool == True:
               wanted_directory_attributes = [main_directory,'Figures',dataset,'SingleEvalPlot'] 
            else:
                wanted_directory_attributes = [main_directory,'Figures',dataset,'DoubleEvalPlot'] 
        results_directory = parse_directory(wanted_directory_attributes)

        
        
        try:
            Path(results_directory).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")

        
        plt.savefig(parse_directory([results_directory,clf_name+'.png']),bbox_inches='tight')


"""config_list = [ dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RF',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'GP',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RS',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                ,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_RF',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_GP',results_type = 'Metric', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]
config_results =  [get_results_per_optimizer(config) for config in config_list]






surrogate_config_list = [dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RF',results_type = 'Surrogate_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'GP',results_type = 'Surrogate_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RS',results_type = 'Surrogate_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_RF',results_type = 'Surrogate_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_GP',results_type = 'Surrogate_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

surrogate_config_results =  [get_results_per_optimizer(config,accumulate=False) for config in surrogate_config_list]


objective_config_list = [dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RF',results_type = 'Objective_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'GP',results_type = 'Objective_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RS',results_type = 'Objective_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_RF',results_type = 'Objective_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_GP',results_type = 'Objective_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

objective_config_results =  [get_results_per_optimizer(config,accumulate=False) for config in objective_config_list]


Acquisition_config_list = [dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RF',results_type = 'Acquisition_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
               #,dict( data_repo = data_repo, seeds = n_seeds , optimizer_type = 'GP',results_type = 'Acquisition_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'RS',results_type = 'Acquisition_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_RF',results_type = 'Acquisition_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                #,dict(data_repo = data_repo, seeds = n_seeds , optimizer_type = 'HEBO_GP',results_type = 'Acquisition_Time', classifier = 'XGB', result_space = 'Single_Space_Results')
                ]

acquisition_config_results =  [get_results_per_optimizer(config,accumulate=False) for config in Acquisition_config_list]
"""


#Get the mean  on all datasets, all optimizers
#Get the min mean per dataset
#  normalize every optimizer on each dataset with max of dataset. 
# get the confidence using interval bla bla

#ylabel (Average normalized 1-AUC)
#xlabel (Number of eval)
"""
datasets_list_run = config_results[0].keys()

#Take the datasets--Dirty.
for dataset in datasets_list_run:
    title_name = get_dataset_name(dataset,config_list[0])
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    
    for config_result_idx in range(len(config_results)):
        opt = config_list[config_result_idx]['optimizer_type']
        
        confidence,means = create_plot_per_optimizer(config_results[config_result_idx][dataset])
        eval_range = means.shape[0]
        x = [i+1 for i in range(eval_range)]
        
        ax1.plot(x,means,opt_colors[opt],label=opt)
        ax1.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)
        
        surrogate_confidence,surrogate_means = create_plot_per_optimizer(surrogate_config_results[config_result_idx][dataset])
        ax2.plot(x,surrogate_means,opt_colors[opt],label=opt)
        ax2.fill_between(x, surrogate_confidence.iloc[0,:], surrogate_confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)

        if config_result_idx == 0:
            objective_confidence,objective_means = create_plot_per_optimizer(objective_config_results[config_result_idx][dataset])
            ax2.plot(x,objective_means,'orange',label='Mean-Objective Time')
            acquisition_confidence,acquisition_means = create_plot_per_optimizer(acquisition_config_results[config_result_idx][dataset])
            ax2.plot(x,acquisition_means,'yellow',label='Mean-acquisition Time')

    x_ticks = [i for i in range(eval_range) if i%50==0]
    plt.xticks(x_ticks,x_ticks)
    plt.xlim([1,eval_range])
    plt.axvline(x = 50, color = 'black', linestyle = '--',label='Initial_Evaluations')
    fig.suptitle(title_name + " /w classifier "  + config_list[0]['classifier'])
    ax1.set_ylabel('(1-AUC)')
    ax2.set_ylabel('Time in seconds')
    plt.xlabel('Number of Objective Evaluations')
    plt.legend(bbox_to_anchor=(1.05, 1),loc = 'upper left')
    #plt.show()
    
    main_directory =  getcwd().replace('\\Dataset_Scripts','')
    wanted_directory_attributes = [main_directory,'Figures',dataset]
    results_directory = parse_directory(wanted_directory_attributes)
    
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    plt.savefig(results_directory+directory_notation+config_list[0]['classifier']+'.png',bbox_inches='tight')
"""




"""
for config_result_idx in range(len(config_results)):
    means_total = []
    for dataset in datasets_list_run:
        opt = config_list[config_result_idx]['optimizer_type']
        confidence,means = create_plot_per_optimizer(config_results[config_result_idx][dataset])
        means_total.append(means)
    means_total_Dataframe = pd.concat(means_total, axis = 1)
    a,b=stats.norm.interval(0.95, loc=means_total_Dataframe.mean(axis=1), scale=means_total_Dataframe.std(axis=1)/np.sqrt(means_total_Dataframe.shape[0]))
    
    #plot each means
    eval_range = means.shape[0]
    x = [i+1 for i in range(eval_range)]
    plt.plot(x,means_normalized,opt_colors[opt],label=opt)
    plt.fill_between(x, confidence_normalized.iloc[0,:], confidence_normalized.iloc[1,:], color=opt_colors[opt], alpha=.1)


x_ticks = [i for i in range(100) if i%10==0]
plt.xticks(x_ticks,x_ticks)
plt.xlim([1,100])
plt.axvline(x = 20, color = 'black', linestyle = '--',label='Initial_Evaluations')
plt.title('Average effectiveness of BO methods ' + " /w classifier "  + config_list[0]['classifier'])
plt.ylabel('Average normalized (1-AUC)/min(1-AUC)')
plt.legend()
main_directory =  getcwd().replace('\\Dataset_Scripts','')
wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets']
results_directory = parse_directory(wanted_directory_attributes)
try:
    Path(results_directory).mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print("Folder is already there")
else:
    print("Folder was created")
plt.savefig(results_directory+directory_notation+config_list[0]['classifier']+'.png',bbox_inches='tight')
"""

