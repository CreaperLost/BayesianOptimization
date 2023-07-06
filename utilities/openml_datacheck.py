import openml
from openml import tasks, runs
import pandas as pd
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

new_df = pd.DataFrame(rows).to_csv('datasets_openml.csv')





benchmark = openml.study.get_suite(218)
task_data = tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)

keep_these = ['tid', 'did', 'name', 'NumberOfFeatures', 'NumberOfInstances',
       'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
       'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures',
       'MaxNominalAttDistinctValues']
task_data[keep_these].to_csv('AutoMLbenchmark.csv')

"""# List all tasks in a benchmark
"""


"""print()

# Return benchmark results
df=openml.evaluations.list_evaluations(
    function="area_under_roc_curve", 
    tasks=benchmark.tasks, 
    output_format="dataframe"
)   # .to_csv('tasks.csv')


print(df)"""