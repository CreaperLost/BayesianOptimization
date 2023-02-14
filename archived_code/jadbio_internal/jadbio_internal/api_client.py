import json
import time
import requests
from typing import List
from requests_toolbelt.multipart.encoder import MultipartEncoder
from jadbio_internal.api.image_client import ImageClient
from jadbio_internal.api.ml_client import MLClient
from jadbio_internal.api.project_client import ProjectClient


class JADBioError(Exception):
    msg = None

    def __init__(self, message):
        self.msg = message


class ApiClient:
    host = None
    token = None
    image: ImageClient = None
    ml: MLClient = None
    project: ProjectClient = None

    def __init__(self, host: str, user: str, passwd: str):
        """
        :param host: Api endpoint
        :param user: username
        :param passwd: user password
        """
        self.host = host
        self.token = self.__login(host, user, passwd)
        self.image = ImageClient(host, self.token)
        self.ml = MLClient(host, self.token)
        self.project = ProjectClient(host, self.token)

    def load_project(self, project_name: str):
        """
        :param project_name: the name of the project
        :return: project id, if project exists
        """
        resp = requests.post(
            self.host + '/api/project/searchByName',
            data=json.dumps({'query': project_name}),
            headers=self.__headers(),
            verify=True
        )
        if not resp.ok:
            print(resp.content)
            raise JADBioError
        json_body = json.loads(resp.content)
        return json_body['id']

    def create_project(self, project_name: str):
        """
        :param project_name:  Name of the new project
        :return: new project id
        """
        resp = requests.post(
            self.host + '/api/project/newProject',
            data=json.dumps({'title': project_name, 'type': 'General'}),
            headers=self.__headers(),
            verify=True
        )
        if not resp.ok:
            raise
        json_body = json.loads(resp.content)
        return json_body['id']

    def load_dataset(self, project, dataset_name):
        """
        :param project: id of the project (returned by load_project, or create project)
        :param dataset_name: name of the dataset
        :return: dataset id if dataset exists
        """
        resp = requests.post(
            self.host + '/api/project/searchDataArray',
            data=json.dumps({'pid': project, 'query': dataset_name}),
            headers=self.__headers(),
            verify=True
        )
        if not resp.ok:
            raise 'Error'
        json_body = json.loads(resp.content)
        if json_body['status'] == 'ERROR':
            raise JADBioError('Dataset not found')
        return json_body['id']

    def delete_dataset(self, dataset):
        resp = requests.post(
            self.host + '/api/project/deleteProjectDataArray',
            data=json.dumps({'id': dataset}),
            headers=self.__headers(),
            verify=True
        )
        if not resp.ok:
            raise

    def upload_dataset_if_not_exists(self, project, dataset_name, dataset_path):
        try:
            return self.load_dataset(project, dataset_name)
        except JADBioError:
            return self.upload_dataset(project, dataset_name, dataset_path)

    def upload_dataset(self, project, dataset_name, dataset_path):
        """
        :param project: id of the project (returned by load_project, or create project)
        :param dataset_name: name of the dataset
        :param dataset_path: location of the dataset
        :return:
        """
        mp_encoder = MultipartEncoder(
            fields={
                'pid': str(project),
                'attributes': json.dumps({
                    'name': dataset_name,
                    'type': 'General',
                    'attributes': {'separator': ',', 'row_store': True, 'has_snames': True, 'has_fnames': True}
                }),
                'file': ('file.txt', open(dataset_path, 'rb'), 'text/plain'),
            }
        )
        resp = requests.post(
            self.host + "/api/dataArray/upload",
            data=mp_encoder,
            headers={
                'Content-Type': mp_encoder.content_type,
                'Authorization': "Bearer" + " " + self.token
            },
            verify=True
        )

        if not resp.ok:
            print(resp.content)
            raise
        tid = json.loads(resp.content)['id']
        return self.__wait_for(tid)

    def submit_batch_analysis(
            self,
            target_dataset: int,
            train_data: List[int],
            exponential: bool,
            tuning: str,
            parallelism: int,
            perform_feature_selection: bool = True
    ):
        """
        :param target_dataset: id of the dataset containing the target features 
        :param train_data: id list of the train datasets
        :param exponential: 
            if false, then an analysis will be created for each combination of
            (target feature, train dataset), along with an analysis containing all train datasets combined
            else if true, all powerset combinations of the train datasets will be created, and
            an analysis will be created for each pair(target feature, powerset combination)
            e.g if target_dataset contains ['t1','t2'] features, and train data is [1,2,3]
            if exponential the following analyses will be created
                ('t1',1),('t1',2),('t1',3),('t2',1),('t2',2),('t2',3),('t1',[1,2,3]),('t2',[1,2,3])
            else if not exponential the following analyses will be created
                ('t1',1),('t1',2),('t1',3),('t1',[1,2]),('t1',[2,3]),('t1',[1,3]),('t1',[1,2,3])
                ('t2',1),('t2',2),('t2',3),('t2',[1,2]),('t2',[2,3]),('t2',[1,3]),('t2',[1,2,3])

        :param tuning: QUICK, NORMAL or EXTENSIVE
        :param parallelism: number of cores for each analysis
        :param perform_feature_selection: 
        :return: 
        """""
        resp = requests.post(
            self.host + '/api/analysis/createBatchAnalysis',
            data=json.dumps({
                'targetPdaId': target_dataset,
                'trainPdaIds': train_data,
                'exponential': exponential,
                'analysisForm': {
                    'tuningEffort': tuning,
                    'accountUtilization': parallelism,
                    'fsPreference': perform_feature_selection,
                    'onlyInterpretable': False
                }
            }),
            headers=self.__headers(),
            verify=True
        )
        if not resp.ok:
            raise
        r = json.loads(resp.content)
        print('Analyses created: {}'.format(r['analysesCreated']))
        print('Warnings: {}'.format(r['warnings']))

    def __authorization_header(self):
        return {'Authorization': 'Bearer' + ' ' + self.token}

    def __headers(self):
        return {'Content-type': 'application/json', 'Authorization': 'Bearer' + ' ' + self.token}

    def __login(self, host, user, passw):
        login_resp = requests.post(
            host + "/api/auth/signin",
            data=json.dumps({'usernameOrEmail': user, 'password': passw}),
            headers={'Content-type': 'application/json'}, verify=True
        )
        if not login_resp.ok:
            raise
        json_body = json.loads(login_resp.content)
        return json_body['accessToken']

    def __wait_for(self, task_id):
        while True:
            print('waiting for task ' + str(task_id))
            resp = requests.post(
                self.host + "/api/dataArray/getTaskState",
                headers=self.__headers(),
                data=json.dumps({'id': task_id}),
                verify=True
            )
            if not resp.ok:
                print(resp.content)
                raise
            json_body = json.loads(resp.content)
            if (json_body['state'] == 'FINISHED'):
                return json_body['did']
            time.sleep(1)
