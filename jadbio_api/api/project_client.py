import requests
import json

import sys
sys.path.insert(1, '..')
from .auth import json_headers


class ProjectClient:
    host: str = None
    token: str = None

    def __init__(self, host: str, token: str):
        self.host = host
        self.token = token

    def find_project(self, name: str):
        resp = requests.post(
            self.host + '/api/project/searchByName',
            data=json.dumps({'query': name}),
            headers= json_headers(self.token)
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)['id']

    def delete_project(self, id: int):
        resp = requests.post(
            self.host + '/api/project/delete',
            data=json.dumps({'id': id}),
            headers= json_headers(self.token)
        )
        if not resp.ok:
            print(resp.content)
            raise

    def find_dataset(self, project: int, name: str):
        resp = requests.post(
            self.host + '/api/project/searchDataArray',
            data=json.dumps({'pid': project, 'query': name}),
            headers= json_headers(self.token)
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)['id']

    def find_project_datasets(self, project: int):
        resp = requests.post(
            self.host + '/api/project/getProjectDataArrays',
            data=json.dumps({'id': project}),
            headers=json_headers(self.token)
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)

    def download_dataset(self, dataset_id: int, download_path: str):
        resp = requests.post(
            self.host + '/api/dataArray/testing/downloadDataset',
            data=json.dumps({'id': dataset_id}),
            headers=json_headers(self.token)
        )
        print(type(resp.content))
        if not resp.ok:
            print(resp.content)
            raise
        open(download_path, 'wb').write(resp.content)


    def find_project_dataset_feature(self, dataset_id: int, feature: str):
        resp = requests.post(
            self.host + '/api/project/dataset/featureInfo',
            data=json.dumps({'pdaId': dataset_id, 'feature': feature}),
            headers=json_headers(self.token)
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)