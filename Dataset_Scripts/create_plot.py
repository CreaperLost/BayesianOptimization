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
def get_results_per_optimizer(config={},accumulate='none'):
    assert config!={}
    assert accumulate == 'none' or accumulate =='min' or accumulate =='addition'

    #Get the configurations partials.
    result_space, classifier ,results_type ,optimizer_type, number_of_seeds, data_repo  = break_config_into_pieces_for_plots(config)

    #Get us the main file
    main_directory =  getcwd().replace(directory_notation+'Dataset_Scripts','')

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
            elif accumulate == 'addition':
                accumulation = np.cumsum(opt_results)
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

    #print(path)

    

    if data_repo == 'Jad':
        path = parse_directory([main_directory,'Jad_Full_List.csv'])
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


#Essential after initial_configs propagate the cost by the corresponding step.
# For batch acquisition function
def propagate_batch(time_evals:pd.DataFrame,step = 10,initial_config = 20):
    l_eval = time_evals.to_list()
    extra_l=list(itertools.chain.from_iterable(itertools.repeat(x, step) for x in l_eval[initial_config:]))
    full_eval=l_eval[:initial_config] + extra_l
    #print(full_eval)
    return full_eval

def get_Jad_avg_score(dataset_name):
    parent_dir = os.path.join(os.getcwd(), os.pardir)
    Res_File = pd.read_csv(os.path.join(parent_dir,'JAD_Results_AUC.csv'),index_col=0).set_index('dataset')
    if dataset_name not in Res_File.index:
        return None
    return Res_File.loc[dataset_name].values[0]

def plot_per_dataset(config):
    clf_name = config['classifier']
    result_space = config['result_space']
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
            config_dict_per_metric=dict({ 'data_repo' : data_repo, 'seeds' : seeds , 'optimizer_type' : opt_name,'results_type' : metric, 'classifier' : clf_name, 'result_space' : result_space})
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
        elif metric_config_name == 'Total_Time':
            acc= 'addition'
        else:
            acc= 'none'
        configuration_results[metric_config_name] = [get_results_per_optimizer(config,acc) for config in configuration_list_for_a_metric]

    #Get the list of datasets run on the metrics.
    datasets_list_run = configuration_results[metric_list[0]][0].keys()

    #Take the datasets--Dirty.
    for dataset in datasets_list_run:
        #Inserting the first of many configurations give_us acess to many of the fields we want. -- Dirty.
        
        title_name = get_dataset_name(dataset,total_config_dictionary[metric_list[0]][0])

        jad_score =  get_Jad_avg_score(title_name)
        


        if double_plot_bool == True:
            fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        else: 
            fig, ax1 = plt.subplots()
        #for each configuration in the configurations do:
        #At this point we essentially parse throught the optimizers.
        for config_result_idx in range(len(configuration_results[metric_list[0]])):
            if time_plot_bool == 'True':

                #The configuration list of results for the specific metric
                config_list_score = configuration_results['Metric']
                config_list_time = configuration_results['Total_Time']

                # The configuration list of optimizers used for specific metric
                config_list_per_optimizer = total_config_dictionary['Metric']

    
                #Get the optimizers.
                opt = config_list_per_optimizer[config_result_idx]['optimizer_type']
                
                

                #Get the confidence and means for the specific dataset.
                confidence,means = create_plot_per_optimizer(config_list_score[config_result_idx][dataset])
                #Get the confidence in time and the mean metric for the specific dataset
                time_confidence,time_means = create_plot_per_optimizer(config_list_time[config_result_idx][dataset])
                if opt == 'HEBO_RF5':
                    time_means= propagate_batch(time_means,5,interval)
                elif opt == 'HEBO_RF10':
                    time_means= propagate_batch(time_means,10,interval)
                x = config_list_time[config_result_idx][dataset]
                ax1.plot(time_means,means,opt_colors[opt],label=opt)
                ax1.fill_between(time_means, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)
                ax1.set_xlim(left=0)
                ax1.set_xlabel('Total Time in Seconds')
            else:
                for metric_measured in metric_list: 
                    
                    #The configuration list of results for the specific metric
                    config_list_per_metric = configuration_results[metric_measured]
                    # The configuration list of optimizers used for specific metric
                    config_list_per_optimizer = total_config_dictionary[metric_measured]

                    #Get the optimizer
                    opt = config_list_per_optimizer[config_result_idx]['optimizer_type']

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
                            #print(x,means)
                            ax2.plot(x,means,opt_colors[opt],label=opt)
                            ax2.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)
                
                #Initial configurations.    
        if time_plot_bool == 'False':
            x_ticks = [i for i in range(eval_range) if i%interval==0]
            plt.xticks(x_ticks,x_ticks)
            plt.xlim([1,eval_range])
            plt.axvline(x = interval, color = 'black', linestyle = '--',label='Initial_Evaluations')
            plt.xlabel('Number of Objective Evaluations')

            if double_plot_bool == True:
                ax2.set_ylabel('Time in seconds')

        if  jad_score != None:
            plt.axhline(y=1-jad_score,color = 'black', linestyle = '-',label='Jad Score')
        # This happens for all the figures.
        ax1.set_ylabel('(1-AUC)')
        fig.suptitle(title_name + " /w classifier "  + clf_name)
        plt.legend(bbox_to_anchor=(1.05, 1),loc = 'upper left')
                

        main_directory =  getcwd().replace(directory_notation+'Dataset_Scripts','')
        if time_plot_bool == 'True':
            wanted_directory_attributes = [main_directory,'Figures',data_repo,dataset,'TimePlot']
        elif time_plot_bool == 'False':
            if double_plot_bool == False:
               wanted_directory_attributes = [main_directory,'Figures',data_repo,dataset,'SingleEvalPlot'] 
            else:
                wanted_directory_attributes = [main_directory,'Figures',data_repo,dataset,'DoubleEvalPlot'] 
        results_directory = parse_directory(wanted_directory_attributes)

        
        
        try:
            Path(results_directory).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
            #print("Folder is already there")
        else:
            #print("Folder was created")
            pass
        
        plt.savefig(parse_directory([results_directory,clf_name+'.png']),bbox_inches='tight')





def plot_average(config):

    clf_name = config['classifier']
    result_space = config['result_space']
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
            config_dict_per_metric=dict({ 'data_repo' : data_repo, 'seeds' : seeds , 'optimizer_type' : opt_name,'results_type' : metric, 'classifier' : clf_name, 'result_space' : result_space})
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
        elif metric_config_name == 'Total_Time':
            acc= 'addition'
        else:
            acc= 'none'
        configuration_results[metric_config_name] = [get_results_per_optimizer(config,acc) for config in configuration_list_for_a_metric]

    #Get the list of datasets run on the metrics.
    datasets_list_run = configuration_results[metric_list[0]][0].keys()


    for config_result_idx in range(len(configuration_results['Metric'])):
        means_total = []
        means_time_total = []
        for dataset in datasets_list_run:
            opt = total_config_dictionary['Metric'][config_result_idx]['optimizer_type']
            #Get the mean of the metric per dataset per optimizer.
            confidence,means = create_plot_per_optimizer(configuration_results['Metric'][config_result_idx][dataset])
            means_total.append(means)
            confidence_time,means_time = create_plot_per_optimizer(configuration_results['Total_Time'][config_result_idx][dataset])
            means_time_total.append(means_time)



        means_total_Dataframe = pd.concat(means_total, axis = 1)
        
        a,b=stats.norm.interval(0.95, loc=means_total_Dataframe.mean(axis=1), scale=means_total_Dataframe.std(axis=1)/np.sqrt(means_total_Dataframe.shape[0]))
        
        confidence_normalized,means_normalized = pd.DataFrame(np.vstack((a,b))) , means_total_Dataframe.mean(axis=1)

        if time_plot_bool == 'False':
            #plot each means
            eval_range = means.shape[0]
            x = [i+1 for i in range(eval_range)]
        elif time_plot_bool == 'True':
            time_mean = pd.concat(means_time_total, axis = 1).mean(axis=1)
            x = time_mean

            #print(opt, x)

            if opt == 'HEBO_RF5':
                x= propagate_batch(x,5,interval)
            elif opt == 'HEBO_RF10':
                x= propagate_batch(x,10,interval)
        #print(opt,means_normalized.iloc[0:100:30],means_normalized.iloc[19])
        plt.plot(x,means_normalized,opt_colors[opt],label=opt)
        plt.fill_between(x, confidence_normalized.iloc[0,:], confidence_normalized.iloc[1,:], color=opt_colors[opt], alpha=.1)

    if time_plot_bool == 'False':
        x_ticks = [i for i in range(eval_range) if i%interval==0]
        plt.xticks(x_ticks,x_ticks)
        plt.xlim([1,eval_range])
        plt.axvline(x = interval, color = 'black', linestyle = '--',label='Initial_Evaluations')
        plt.xlabel('Number of objective evals.')
    elif time_plot_bool =='True':
        plt.xlim(left=0)
        plt.xlabel('Average time in seconds')


    plt.title('Average effectiveness of BO methods ' + " /w classifier "  + clf_name)
    plt.ylabel('Average `1-AUC')
    plt.legend()
    main_directory =  getcwd().replace(directory_notation+'Dataset_Scripts','')

    if time_plot_bool == 'True':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','TimePlot']
    elif time_plot_bool == 'False':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','SingleEvalPlot'] 
    elif time_plot_bool =='Partial':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','PartialEvalPlot'] 
            
    results_directory = parse_directory(wanted_directory_attributes)
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    plt.savefig(parse_directory([results_directory,clf_name+'.png']),bbox_inches='tight')

#Get the mean  on all datasets, all optimizers
#Get the min mean per dataset
#  normalize every optimizer on each dataset with max of dataset. 
# get the confidence using interval bla bla

#ylabel (Average normalized 1-AUC)
#xlabel (Number of eval)
def plot_two_categories(data1,data2,opt_list,clf_name,time_plot_bool,time_data1=None,time_data2=None):
    for opt_index in range(len(data1)):
        opt = opt_list[opt_index]
        total_data = pd.concat((data1[opt_index],data2[opt_index]),axis=1)
        
        #print(total_data)
        a,b=stats.norm.interval(0.95, loc=total_data.mean(axis=1), scale=total_data.std(axis=1)/np.sqrt(total_data.shape[0]))
        confidence_normalized,means_normalized = pd.DataFrame(np.vstack((a,b))) , total_data.mean(axis=1)

        if time_plot_bool == 'False':
            #plot each means
            eval_range = means_normalized.shape[0]
            x = [i+1 for i in range(eval_range)]
        elif time_plot_bool == 'True':
            time_mean = pd.concat((time_data1[opt_index],time_data2[opt_index]),axis=1).mean(axis=1)
            x = time_mean

            """if opt == 'HEBO_RF5':
                x= propagate_batch(x,5,interval)
            elif opt == 'HEBO_RF10':
                x= propagate_batch(x,10,interval)"""
        print(opt,means_normalized.iloc[199])
        plt.plot(x,means_normalized,opt_colors[opt],label=opt)
        plt.fill_between(x, confidence_normalized.iloc[0,:], confidence_normalized.iloc[1,:], color=opt_colors[opt], alpha=.1)

    if time_plot_bool == 'False':
        x_ticks = [i for i in range(eval_range) if i%interval==0]
        plt.xticks(x_ticks,x_ticks)
        plt.xlim([1,eval_range])
        #plt.axvline(x = 10, color = 'black', linestyle = '--',label=' 10 Initial_Evaluations')
        plt.axvline(x = 20, color = 'black', linestyle = '--',label=' 20 Initial_Evaluations')
        #plt.axvline(x = 50, color = 'black', linestyle = '--',label=' 50 Initial_Evaluations')
        plt.xlabel('Number of objective evals.')
    elif time_plot_bool =='True':
        plt.xlim(left=0)
        plt.xlabel('Average time in seconds')


    plt.title('Average effectiveness of BO methods ' + " /w classifier "  + clf_name)
    plt.ylabel('Average `1-AUC')
    plt.legend()
    main_directory =  getcwd().replace(directory_notation+'Dataset_Scripts','')

    if time_plot_bool == 'True':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','TimePlot']
    elif time_plot_bool == 'False':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','SingleEvalPlot'] 
    elif time_plot_bool =='Partial':
        wanted_directory_attributes = [main_directory,'Figures','OverAllDatasets','PartialEvalPlot'] 
            
    results_directory = parse_directory(wanted_directory_attributes)
    try:
        Path(results_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    plt.savefig(parse_directory([results_directory,clf_name+'.png']),bbox_inches='tight')

def get_average_per_category(config):
    clf_name = config['classifier']
    result_space = config['result_space']
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
            config_dict_per_metric=dict({ 'data_repo' : data_repo, 'seeds' : seeds , 'optimizer_type' : opt_name,'results_type' : metric, 'classifier' : clf_name, 'result_space' : result_space})
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
        elif metric_config_name == 'Total_Time':
            acc= 'addition'
        else:
            acc= 'none'
        configuration_results[metric_config_name] = [get_results_per_optimizer(config,acc) for config in configuration_list_for_a_metric]

    #Get the list of datasets run on the metrics.
    datasets_list_run = configuration_results[metric_list[0]][0].keys()
    
    
    per_opt = []
    per_opt_time = []
    for config_result_idx in range(len(configuration_results['Metric'])):
        means_total = []
        means_time_total = []
        for dataset in datasets_list_run:
            opt = total_config_dictionary['Metric'][config_result_idx]['optimizer_type']
            #Get the mean of the metric per dataset per optimizer.
            confidence,means = create_plot_per_optimizer(configuration_results['Metric'][config_result_idx][dataset])
            means_total.append(means)
            confidence_time,means_time = create_plot_per_optimizer(configuration_results['Total_Time'][config_result_idx][dataset])
            means_time_total.append(means_time)

        

        means_total_Dataframe = pd.concat(means_total, axis = 1)
        means_time_total_Dataframe = pd.concat(means_time_total, axis = 1)
        per_opt.append(means_total_Dataframe)
        per_opt_time.append(means_time_total_Dataframe)
        #a,b=stats.norm.interval(0.95, loc=means_total_Dataframe.mean(axis=1), scale=means_total_Dataframe.std(axis=1)/np.sqrt(means_total_Dataframe.shape[0]))
        
        #confidence_normalized,means_normalized = pd.DataFrame(np.vstack((a,b))) , means_total_Dataframe.mean(axis=1)


    return per_opt,per_opt_time


colors = ['red','blue','green','black','purple','orange','grey','cyan','yellow']

"""opt_colors= {
    'RS':colors[0],
    'RF':colors[1],
    'GP':'green',
    'HEBO_RF':'black',
    'HEBO_GP':'purple',
    'HEBO_RF5':'orange',
    'HEBO_RF10':'grey',
    'Sobol':'purple',
    'HEBO_RF_ACQ100':'orange',
    'HEBO_RF_ACQ500':'blue',
    'HEBO_RF_Scipy':'cyan',
    'HEBO_RF_DE':'orange',
    'GP_INIT10' : 'orange',
    'GP_INIT50' : 'cyan',
    'HEBO_RF_INIT10':'grey',
    'HEBO_RF_INIT50':'blue',
}"""



data_repo = 'Jad'
n_seeds=  3
metrics = ['Metric','Surrogate_Time','Objective_Time','Acquisition_Time','Total_Time']
time_plot = True
double_plot = False
#How many initial configurations we have run.
interval = 20
result_space = 'Trees_Single_Space_Results'
optimizers = ['HEBO_RF_NTREE_500','HEBO_RF_NTREE_500_ACQ10000'] 

opt_colors = dict()
clr_pos = 0
for opt in optimizers:
    opt_colors.update({opt:colors[clr_pos]})
    clr_pos+=1


for data_repo in ['Jad','OpenML']:
    

    for bool_flag in ['False','True']:
        time_plot = bool_flag
        double_plot = False

        general_config = {
        'classifier':'XGB',
        'result_space':result_space,
        'optimizers' : optimizers,
        'n_seeds' : n_seeds,
        'data_repo':data_repo,
        'double_plot':double_plot,
        'metrics': metrics,
        'time_plot':time_plot,
        }
    

        plot_per_dataset(general_config)
        plt.clf()

"""  
for bool_flag in ['False','True']:
    means_per_cat = []
    means_per_cat_time = []
    for data_repo in ['Jad','OpenML']:
    
        time_plot = bool_flag
        double_plot = False

        general_config = {
        'classifier':'XGB',
        'result_space':result_space,
        'optimizers' : optimizers,
        'n_seeds' : n_seeds,
        'data_repo':data_repo,
        'double_plot':double_plot,
        'metrics': metrics,
        'time_plot':time_plot,
        }
    

        #plot_per_dataset(general_config)
        opt, opt_time  = get_average_per_category(general_config)
        print('Time',opt)
        means_per_cat.append( opt )
        means_per_cat_time.append( opt_time )
    plt.clf()
    #One category is for JAD, the other is for OpenML.
    plot_two_categories(means_per_cat[0],means_per_cat[1],optimizers,'XGB',bool_flag,means_per_cat_time[0],means_per_cat_time[1])
"""  

    