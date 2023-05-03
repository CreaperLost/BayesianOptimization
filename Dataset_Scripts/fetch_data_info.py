from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
from dataset_utils import get_dataset_ids,get_data_list,filter_datasets,save_info_simple

data_id = [851,842,1114,839,847,850,866,883,843]
data_list  = pd.read_csv('Jad_Full_List.csv')   #get_data_list('Jad')
    
#data_list.to_csv('Jad_Full_List.csv')
print(data_id)
remaining_data =filter_datasets(datalist=data_list,data_ids=data_id,repo = 'Jad') 

print(remaining_data)
save_info_simple(remaining_data,'Jad')



data_id = [11,14954,43,3021,3917,3918,9952,167141,167125,9976,2074,9910]
#data_list  = get_data_list('OpenML')
data_list  = pd.read_csv('Openmlcc18.csv',header=0,index_col=0)   
#data_list.to_csv('OpenML_Full_List.csv')
data_id = list( data_list[data_list['tid'].isin(data_id)]['did'] )
print(data_id)
remaining_data =filter_datasets(datalist=data_list,data_ids=data_id,repo = 'OpenML') 
save_info_simple(remaining_data,'OpenML') 



jad_data = pd.read_csv('Jad_dataset_characteristics.csv')
openml_data = pd.read_csv('OpenML_dataset_characteristics.csv')
names_list = list(jad_data['name'].values) + list(openml_data['name'].values)
