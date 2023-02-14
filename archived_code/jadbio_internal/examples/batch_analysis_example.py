from jadbio_internal.api_client import ApiClient

client = ApiClient("https://api.jadbio.com", "username", "password")

#if you already have a project and want to load it
project = client.load_project('api')

#or if you want to create a new project
# project = jadbio_internal.create_project('api_test')

#if you have already uploaded data in the project use load_project
# alz = jadbio_internal.load_dataset(project, 'Alzheimer')


#or if you want to upload a dataset user upload_dataset
alz_target = client.upload_dataset(project, 'AlzheimerTarget', '/home/p01/data/datasets/batch/Alzheimer_target2.csv')
alz_train1 = client.upload_dataset(project, 'AlzheimerTrain1', '/home/p01/data/datasets/batch/Alzheimer_train1.csv')
alz_train2 = client.upload_dataset(project, 'AlzheimerTrain2', '/home/p01/data/datasets/batch/Alzheimer_train2.csv')
#
client.submit_batch_analysis(alz_target, [alz_train1, alz_train2], False, 'NORMAL', 2, True)