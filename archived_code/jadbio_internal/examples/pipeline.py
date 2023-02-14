from jadbio_internal.api_client import ApiClient
import numpy as np
import pandas as pd
# # # Some parameters # # #
# Dataset sample and feature size range
analysis_type = 'CLASSIFICATION'
min_samples = 50
max_samples = 200
min_features = 10
max_features = 100


# # # Select dataset # # #
# Initialise client
jad = ApiClient('https://exp.jadbio.com:4443', 'pkatsogr@gmail.com', '22222222')
project = jad.project.find_project('jad_research')
# Load dataset list
dataset_list = jad.project.find_project_datasets(project)
if analysis_type == 'CLASSIFICATION':
    dataset_list = [d for d in dataset_list
                    if jad.project.find_project_dataset_feature(d['id'], 'target')['type'] == 'Categorical']
else:
    dataset_list = [d for d in dataset_list
                    if jad.project.find_project_dataset_feature(d['id'], 'target')['type'] == 'Numerical']
dataset_list = [dataset_list[d] for d in range(len(dataset_list))
                if (dataset_list[d]['samples'] >= min_samples) & (dataset_list[d]['samples'] <= max_samples) &
                (dataset_list[d]['features'] >= min_features) & (dataset_list[d]['features'] <= max_features)]
df = pd.DataFrame(dataset_list,columns=list(dataset_list[0].keys()))
print(df)

df.to_csv('Dataset_List.csv',index=None)  