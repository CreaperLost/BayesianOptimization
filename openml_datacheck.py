import openml
from openml import tasks, runs
# List all datasets and their properties
# 
# 
datasets =openml.datasets.list_datasets(output_format="dataframe",status='active')
print(datasets)

# List all tasks in a benchmark
benchmark = openml.study.get_suite('OpenML-CC18')
tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks).to_csv('Openmlcc18.csv')


print()

# Return benchmark results
openml.evaluations.list_evaluations(
    function="area_under_roc_curve", 
    tasks=benchmark.tasks, 
    output_format="dataframe"
).to_csv('tasks.csv')