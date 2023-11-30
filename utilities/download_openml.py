import openml
from openml import tasks, runs
import pandas as pd
import os


# Get all openML data here.
datasets =openml.datasets.list_datasets(output_format="dataframe",status='active')

# Choose a specific Benchmark 
# You can use a INTEGER here instead of a string
benchmark = openml.study.get_suite('OpenML-CC18')

# Get all tasks for this benchmark.
task_data = tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks) 

# Use this to get a csv about the information of the data.
#task.to_csv('Openmlcc18.csv')

# Put the OpenML IDs you want to download.
# Careful. Here you put the Task ID.
OpenML_ids=[11,14954,43,3021,3917,3918,9910,9952,9976,167125,167141,2074]
rows = []
for index, row in task_data.iterrows():
    if row['tid'] in OpenML_ids:
        rows.append(row)
        # Get the dataset for the task.
        d = openml.datasets.get_dataset(row['did'], download_data=False)
        
        # X : Features, Y: Label
        # categorical : which are categorical {False,True}
        # attribute_names : which are column names {X: feature names}
        X, y, categorical, attribute_names = d.get_data(target=row['target_feature'])

        features = pd.DataFrame(X,columns= attribute_names)
        data=pd.concat((features,y),axis=1)

        #Save the data in a csv by name.
        data.to_csv(row['name']+'.csv')
        