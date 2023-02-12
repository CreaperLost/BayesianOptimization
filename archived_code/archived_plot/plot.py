from os import listdir,getcwd
from os.path import isfile, join,isdir
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import openml
 



parent_path = getcwd() + '\Results'
Dataset_files = [f for f in listdir(parent_path) if isdir(join(parent_path, f))]

n_seeds = 5

dataset_dict = dict()
for dataset in  Dataset_files:
    seed_path = parent_path + '/' + dataset
    seed_files = [f for f in listdir(seed_path) if isdir(join(seed_path, f))]
    
    RF_seed = []
    RS_seed = []
    GP_seed = []
    for seed in seed_files:
        opt_path = seed_path + '/' + seed
        opt_files = [f for f in listdir(opt_path) if isfile(join(opt_path, f))]

        for optimizer in opt_files:
            #Random Search
            file_path = opt_path + '/' + optimizer

            df=pd.read_csv(file_path,index_col=0)

            accumulation=np.minimum.accumulate(df)
            
            if str(optimizer).split('.')[0] == 'RS' :
                RS_seed.append(accumulation)
            elif str(optimizer).split('.')[0] == 'RF' :
                RF_seed.append(accumulation)
            elif str(optimizer).split('.')[0] == 'GP' :
                GP_seed.append(accumulation)
    if len(RF_seed) < n_seeds or len(RS_seed) < n_seeds or len(GP_seed)< n_seeds:
        continue
    dataset_dict[dataset] = {
        'RF': RF_seed,
        'RS': RS_seed,
        'GP': GP_seed,
    }


#Computes the bounds for each evaluation point, upper and lower confidence. a is the best, b is the worst.
#Returns dataframe. 1 row is the best, 2nd row is the worst.
# df.iloc[0,:] , df.iloc[1,:] to access
# returns also the mean
def get_seeds_per_dataset(seeds):
    df_concat = pd.concat(seeds, axis = 1)
    a,b=stats.norm.interval(0.95, loc=df_concat.mean(axis=1), scale=df_concat.std(axis=1)/np.sqrt(len(seeds)))
    return pd.DataFrame(np.vstack((a,b))) , df_concat.mean(axis=1)

optimizers = ['RS','RF','GP']
opt_colors= {
    'RS':'r',
    'RF':'b',
    'GP':'g',
}

for dataset in Dataset_files:
    task_id = dataset.split('Dataset')[1] 
    task = openml.tasks.get_task(task_id, download_data=False)
    dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    if dataset not in dataset_dict.keys():
        continue
    fig, ax = plt.subplots()
    for opt in optimizers:
        confidence, means = get_seeds_per_dataset(dataset_dict[dataset][opt])
        eval_range = means.shape[0]
        x = [i+1 for i in range(eval_range)]
        
        ax.plot(x,means,opt_colors[opt],label=opt)
        ax.fill_between(x, confidence.iloc[0,:], confidence.iloc[1,:], color=opt_colors[opt], alpha=.1)

    x_ticks = [i for i in range(eval_range) if i%10==0]
    plt.xticks(x_ticks,x_ticks)
    plt.xlim([1,eval_range])
    plt.title(dataset_info.name)
    plt.ylabel('Error Rate (1-Accuracy)')
    plt.legend()
    #plt.show()
    plt.savefig('Figures/Discrete/'+dataset_info.name+'.png')


    