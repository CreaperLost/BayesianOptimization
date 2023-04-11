
import os
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
# Best AUC by analysis

folds = 10
repetitions = 3
results_dir = '../classification_typical/'
analyses = os.listdir(results_dir)
best_auc_table = pd.DataFrame(columns=['dataset', 'best'])

for i in range(0, len(analyses)):
    best_auc_table.at[i, 'dataset'] = analyses[i]
    outcome = pd.read_csv(results_dir + analyses[i] + '/target_data.csv', dtype="Int64")
    split_indices = pd.read_csv(results_dir + analyses[i] + '/split_indices.csv', dtype="Int64")
    split_indices.drop('Unnamed: 0', axis=1, inplace=True)
    
    oos = pd.read_csv(results_dir + analyses[i] + '/oos_0.csv')
    print(oos)
    configuration_performances = np.zeros((len(oos), folds * repetitions))
    for f in range(configuration_performances.shape[1]):
        
        if f % folds == 0:
            oos = pd.read_csv(results_dir + analyses[i] + '/oos_' + str(int(round(f / folds))) + '.csv')
            oos.drop('0', axis=1, inplace=True)
            oos = np.array(oos)
        fold_indices = split_indices.loc[f]
        fold_indices = [fold_indices[fi] for fi in range(len(fold_indices)) if ~fold_indices.isna()[fi]]
        
        for c in range(len(oos)):
            print(oos[c, fold_indices])
            configuration_performances[c, f] = roc_auc_score(np.array(outcome.loc[fold_indices]).astype(bool),
                                                                 oos[c, fold_indices].astype(np.float32))
    best_auc_table.at[i, 'best'] = np.max(np.mean(configuration_performances, axis=1))
    print('Analysis', i+1, 'of', len(analyses))