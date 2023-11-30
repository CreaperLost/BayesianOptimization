import openml 
from openml import tasks, runs
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# List all tasks in a benchmark
cc_18_benchmark = openml.study.get_suite('OpenML-CC18')
tasks_cc_18 = tasks.list_tasks(output_format="dataframe", task_id=cc_18_benchmark.tasks)

print(tasks_cc_18)

automl_benchmark =openml.study.get_suite(218)
automl_tasks = tasks.list_tasks(output_format="dataframe", task_id=automl_benchmark.tasks)

print(automl_tasks)


task_id_cc_18 = list ( tasks_cc_18['tid']  )
task_id_automl = list ( automl_tasks['tid']  )


common_task = [i for i in task_id_automl if i in task_id_cc_18]
print(common_task)
print(len(common_task))


automl_tasks.drop(['evaluation_measures'],inplace=True,axis=1)

all_tasks = pd.concat([tasks_cc_18,automl_tasks],axis=0)

all_tasks.drop_duplicates(inplace=True)



print(all_tasks.columns)

#plt.scatter(all_tasks['NumberOfInstances'],all_tasks['NumberOfFeatures'])
#rslt_df = dataframe[dataframe['Percentage'] > 70] 


# Apply some filtering


# Sample fitering

filtering = 'features'

if filtering == 'samples':
    less_tasks = all_tasks[all_tasks['NumberOfInstances'] <= 10000]
    less_tasks = less_tasks[less_tasks['NumberOfInstances'] >= 500]
    print(less_tasks.shape)
    plt.scatter(less_tasks['NumberOfInstances'],less_tasks['NumberOfFeatures'])
elif filtering == 'features':
    less_tasks = all_tasks[all_tasks['NumberOfInstances'] <= 10000]
    less_tasks = less_tasks[less_tasks['NumberOfInstances'] >= 500]
    #less_tasks = less_tasks[less_tasks['NumberOfFeatures'] >= 10]
    less_tasks = less_tasks[less_tasks['NumberOfFeatures'] <= 1000]
    print(less_tasks.shape)
    plt.scatter(less_tasks['NumberOfInstances'],less_tasks['NumberOfFeatures'])
else:
    plt.scatter(all_tasks['NumberOfInstances'],all_tasks['NumberOfFeatures'])

less_tasks.to_csv('AutoML+CC18.csv')

# Set axis titles
plt.ylabel('#Features')
plt.xlabel('#Samples')

plt.show()


print(list(less_tasks['tid']))


"""

# Check which datasets we already run.

jad_data = pd.read_csv('Jad_dataset_characteristics.csv')
openml_data = pd.read_csv('OpenML_dataset_characteristics.csv')
names_list = list(jad_data['name'].values) + list(openml_data['name'].values)


names_list = [
    'nki70', 'kits-subset', 'christine', 
     'white-clover', 'fl2000', 'lungcancer', 
     'langLog', 'squash-stored', 'meta', 
     'balance-scale', 'spambase', 'satimage', 'sick', 'kc1', 'pc1', 
     'Bioresponse', 'phoneme', 'madelon', 'cylinder-bands', 'Internet-Advertisements', 'churn'
]


for i in names_list:
    if i in list(all_tasks['name']):
        print('Exists', i )
    else:
        print('Doesnt',i)"""