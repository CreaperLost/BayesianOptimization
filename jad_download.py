import numpy as np
import pandas as pd


from jadbio_api.api_client import ApiClient
import os
import matplotlib.pyplot as plt
# # # Some parameters # # #
# Dataset sample and feature size range
analysis_type = 'CLASSIFICATION'
min_samples = 50
max_samples = 200
min_features = 10
max_features = 100


# # # Select dataset # # #
# Initialise client

#Download and save as csv.
"""jad = ApiClient('https://exp.jadbio.com:4443', 'pkatsogr@gmail.com', '22222222')
for i in [1114,865,852,851,850,842]:
    jad.project.download_dataset(i,os.getcwd() + '/Jad_Temp/'+ 'dataset'+ str(i) + '.csv')"""
    

for i in [1114,865,852,851,850,842]:
    dataset = pd.read_csv(os.getcwd() + '/Jad_Temp/'+'dataset'+str(i) +'.csv')
    dataset.drop('gr.gnosisda-1',axis=1,inplace=True)
    X = dataset.drop(['target'],axis=1)
    y = dataset['target']
    cat_idx = [X.columns.get_loc(col) for col in X.select_dtypes(include=['object']).columns.tolist() ]
    print(i)
    print(dataset.select_dtypes(include=['object']).columns.tolist())
    print(cat_idx)
    print(dataset.info())
    print(dataset.nunique())
 