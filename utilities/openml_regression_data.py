import openml


benchmark_suite = openml.study.get_suite(suite_id=269) # obtain the benchmark suite
task_data = openml.tasks.list_tasks(output_format="dataframe", task_id=benchmark_suite.tasks) #.to_csv('OpenML_Regression.csv')

print(task_data.head())
keep_these = ['tid', 'did', 'name', 'NumberOfFeatures', 'NumberOfInstances',
       'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
       'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures',
       'MaxNominalAttDistinctValues']
task_data[keep_these].to_csv('OpenML_Regression_Cleaned.csv')
"""task_id = dataset.split('Dataset')[1] 
task = openml.tasks.get_task(task_id, download_data=False)
dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)"""

"""run_ids = []
for task_id in benchmark_suite.tasks: # iterate over all tasks
    task = openml.tasks.get_task(task_id, download_data=False) # download the OpenML task
    dataset_info = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    
    print(f'Data set: {dataset_info.name}, Task ID : {task_id}')
    print(task)

datasets =openml.datasets.list_datasets(output_format="dataframe",status='active')
"""

