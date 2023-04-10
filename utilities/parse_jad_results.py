import csv 
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#Give the name of the wanted dataset, returns the path to the files.
def get_dataset_path(wanted_dataset_name):
    # Get the parent directory.
    parent_dir = os.path.join(os.getcwd(), os.pardir)

    #Join the paths
    path_to_jad_datasets = os.path.join(parent_dir,'classification_typical')

    for entry in os.scandir(path_to_jad_datasets):
        if entry.is_dir() and wanted_dataset_name == entry.name:
            return os.path.join(path_to_jad_datasets,wanted_dataset_name)
    return None
    

def filter_nan_indices(curr_list):
    assert isinstance(curr_list,list)
    # Find the indices of NaN values in the list
    nan_indices = np.where(np.isnan(curr_list))[0]

    # Filter out the NaN values from the list using list comprehension
    new_list = [value for i, value in enumerate(curr_list) if i not in nan_indices]

    return new_list


def get_dataset_split(wanted_dataset_name,folds=10,n_repeats=3):

    path = get_dataset_path(wanted_dataset_name)

    split_index_file_path = os.path.join(path,'split_indices.csv')
    
    data = pd.read_csv(split_index_file_path,index_col=0)


    per_repeat_folds = []
    for n_repeat in range(n_repeats):
        if n_repeat == 0:
            start_fold = 0
            end_fold = folds
        else:
            start_fold = n_repeat*folds
            end_fold = (n_repeat+1)*folds 

        per_repeat_folds.append([filter_nan_indices(list(data.iloc[fold])) for fold in range(start_fold,end_fold)])

    return per_repeat_folds



#Finds the target for each split.
def get_target_data_per_split(dataset_name,dataset_split):

    path = get_dataset_path(dataset_name)

    split_index_file_path = os.path.join(path,'target_data.csv')
    
    target_data = pd.read_csv(split_index_file_path)

    per_repeat =  []
    for n_repeat in  range(len(dataset_split)):
        per_fold = []
        for n_fold in range(len(dataset_split[n_repeat])):
            split_indexes = dataset_split[n_repeat][n_fold]
            #Get the target value for each split.
            target = target_data.iloc[split_indexes].values.flatten().tolist()
            per_fold.append(target)
        per_repeat.append(per_fold)

    return per_repeat

def get_avg_score(dataset_name,target_data_split):
    #Some information.
    n_repeats = len(target_data_split)
    n_folds = len(target_data_split[0])

    #Get the path.
    path = get_dataset_path(dataset_name)

    total =  pd.DataFrame()
    for n_repeat in  range(n_repeats):
        repeat_oos = os.path.join(path,'oos_'+ str(n_repeat) + '.csv')
        oos_data = pd.read_csv(repeat_oos,index_col=0)

        # Iterate through the DataFrame by index, returns a new pd.series. :)
        # Iterate the 10-fold per configuration.
        avg_score_per_config = []
        for index, row in oos_data.iterrows():
            #print(f"Index: {index}")
            #print(f"Values: {row}")
            
            per_fold = []
            start_fold_idx = 0
            for n_fold in range(n_folds):
                # Get the target for the specific fold.
                target = [int(i) for i in target_data_split[n_repeat][n_fold]]
                
                # Find the end_folx_idx
                end_fold_idx = start_fold_idx + len(target) 

                #Find the predictions of the specific fold.
                predictions = row.iloc[start_fold_idx:end_fold_idx]
                
                # for single class this throws exception
                try:
                    predictions = np.array([eval(s) for s in predictions])
                except:
                    pass
                roc_auc = roc_auc_score(target, predictions,multi_class='ovr')
                print(np.append(predictions,np.array(target).reshape(-1,1),axis=1))
                print(roc_auc)
                quit()
                
                #append
                per_fold.append(roc_auc)
                #Append the already used n_folds
                # Iterate to the next indexes.
                #print(start_fold_idx,end_fold_idx,len(target),roc_auc)
                start_fold_idx = end_fold_idx

            mean_auc_of_config_for_fold = sum(per_fold) / len(per_fold)
            avg_score_per_config.append((index,mean_auc_of_config_for_fold))
        result = pd.DataFrame(avg_score_per_config,columns = ['config','avg_auc_repeat_'+str(n_repeat)])
        result.set_index('config', inplace=True)
        total = pd.concat((total,result),axis=1)
    print(total.describe())
    return total.mean(axis=1)

def best_get_avg(dataset_name,n_folds = 10,n_repeats = 3):
    # Shape (n_repeats , n_folds, splits_for_the_fold)
    dataset_split = get_dataset_split(dataset_name,n_folds,n_repeats)
    
    # Get the target for each split.
    # Shape (n_repeats , n_folds, splits_for_the_fold)
    target_data_split = get_target_data_per_split(dataset_name,dataset_split)

    # Avg AUC per config.
    avg_per_config = get_avg_score(dataset_name,target_data_split)
    #Return the maximum
    return avg_per_config.max()

def get_datasets_of_jad():

    #Jad Dataset
    parent_dir = os.path.join(os.getcwd(), os.pardir)

    #Join the paths
    path_to_jad_datasets = os.path.join(parent_dir,'classification_typical')

    result_per_dataset = []
    for entry in os.scandir(path_to_jad_datasets):
        if entry.is_dir():
            result_per_dataset.append((entry.name,best_get_avg(entry.name,10,3)))

    #pd.DataFrame(result_per_dataset,columns=['dataset','AUC']).to_csv(os.path.join(parent_dir,'JAD_Results_AUC.csv'))

get_datasets_of_jad()