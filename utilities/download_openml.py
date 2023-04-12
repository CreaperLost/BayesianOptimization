import openml
from openml import tasks, runs
import pandas as pd
import os
# List all datasets and their properties
# 
# 
datasets =openml.datasets.list_datasets(output_format="dataframe",status='active')


benchmark = openml.study.get_suite('OpenML-CC18')
task_data = tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks) #.to_csv('Openmlcc18.csv')


OpenML_ids=[11,14954,43,3021,3917,3918,9910,9952,9976,167125,167141,2074]
rows = []
for index, row in task_data.iterrows():
    if row['tid'] in OpenML_ids:
        rows.append(row)
        d = openml.datasets.get_dataset(row['did'], download_data=False)
        # X : Features
        # Y: Label
        # categorical : which are categorical {False,True}
        # attribute_names : which are column names {X: feature names}
        X, y, categorical, attribute_names = d.get_data(target=row['target_feature'])
        features = pd.DataFrame(X,columns= attribute_names)
        data=pd.concat((features,y),axis=1)
        data.to_csv(row['name']+'.csv')
        print(data)