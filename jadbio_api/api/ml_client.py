import json
import requests


import sys
sys.path.insert(1, '..')

from .auth import json_headers
from ..ml.analysis_form import AnalysisForm
import time

from ..ml.custom_cv_form import CVArgs


class MLClient:
    host: str = None
    token: str = None
    ssl_verify: bool

    def __init__(self, host: str, token: str, ssl_verify=True):
        self.host = host
        self.token = token
        self.ssl_verify = ssl_verify

    def analyze(self, dataset_id: int, form: AnalysisForm):
        analysis_id = self.submit_analysis(dataset_id, form)
        self.__wait_for_analysis(analysis_id)
        analysis_res = self.__download_analysis_res(analysis_id)
        analysis_res['analysis_id'] = analysis_id
        return analysis_res

    def analyze_testing(self, dataset_id: int, form: AnalysisForm):
        analysis_id = self.submit_analysis(dataset_id, form)
        self.__wait_for_analysis(analysis_id)
        analysis_res = self.__download_testing_analysis_res(analysis_id)
        analysis_res['analysis_id'] = analysis_id
        return analysis_res

    def delete_analysis(self, analysis_id: int):
        resp = requests.post(
            self.host + '/api/analysis/delete',
            headers=json_headers(self.token),
            data=json.dumps({'id': analysis_id}),
            verify=self.ssl_verify
        )
        if not resp.ok:
            print(resp.content)
            raise

    def cv(self, form: CVArgs):
        analysis_id = self.__submit_cv(form)
        self.__wait_for_cv(analysis_id)
        return self.__cv_results(analysis_id)

    def validate(self, analysis_id: int, dataset_id: int):
        validation_id = self.submit_validation(dataset_id, analysis_id)
        self.__wait_for_validation(validation_id)
        return validation_id

    def submit_analysis(self, dataset_id: int, form: AnalysisForm):
        resp = requests.post(
            self.host + '/api/analysis/createAnalysis',
            headers=json_headers(self.token),
            data=json.dumps({'pdaId': dataset_id, 'form': form.to_dict()}),
            verify=self.ssl_verify
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)['id']

    def submit_validation(self, dataset: int, analysis: int):
        resp = requests.post(
            self.host + '/api/analysis/validate',
            headers=json_headers(self.token),
            data=json.dumps({
                'pdaId': dataset,
                'aid': analysis,
                'model': 'best',
                'sid': 0
            }),
            verify=self.ssl_verify
        )
        if not resp.ok:
            print(resp.content)
            raise
        return json.loads(resp.content)['id']

    def __download_testing_analysis_res(self, analysis_id: int):
        resp = requests.post(
            self.host + '/api/testing/analysis/downloadAnalysisResults',
            headers=json_headers(self.token),
            data=json.dumps({'id': analysis_id}),
        )
        if not resp.ok:
            raise
        return json.loads(resp.content)

    def __download_analysis_res(self, analysis_id: int):
        resp = requests.post(
            self.host + '/api/analysis/downloadAnalysisResults',
            headers=json_headers(self.token),
            data=json.dumps({'id': analysis_id}),
        )
        if not resp.ok:
            raise
        return json.loads(resp.content)

    def __wait_for_analysis(self, task_id: int, silent=False):
        while True:
            resp = requests.post(
                self.host + '/api/analysis/status',
                headers=json_headers(self.token),
                data=json.dumps({'id': task_id}),
            )
            if not resp.ok:
                raise
            resp_body = json.loads(resp.content)
            status = resp_body['status']
            if (status == 'ERROR'):
                print('ERROR')
                raise
            elif (status == 'FINISHED'):
                print('FINISHED 100%')
                return
            else:
                progress = resp_body['totalProgress']
                if not silent:
                    print('{} {:.2f}%'.format(status, float(progress)), end='\r')
                time.sleep(1)
        print('')

    def __wait_for_validation(self, task_id: int, silent=False):
        while True:
            resp = requests.post(
                self.host + '/api/analysis/validationStatus',
                headers=json_headers(self.token),
                data=json.dumps({'id': task_id}),
            )
            if not resp.ok:
                raise
            resp_body = json.loads(resp.content)
            status = resp_body['status']
            if (status == 'ERROR'):
                print('ERROR')
                raise
            elif (status == 'FINISHED'):
                print('FINISHED 100%')
                return
            else:
                progress = resp_body['total_progress']
                if not silent:
                    print('{} {:.2f}%'.format(status, 100 * float(progress)), end='\r')
                time.sleep(1)
        print('')

    def __submit_cv(self, form: CVArgs):
        resp = requests.post(
            self.host + '/api/custom_ml/cv',
            headers=json_headers(self.token),
            data=json.dumps(form.to_dict()),
            verify=self.ssl_verify
        )
        if not resp.ok:
            print(resp.content)
            raise
        return int(json.loads(resp.content)['payload'])

    def __wait_for_cv(self, task_id: int, silent=False):
        while True:
            resp = requests.post(
                self.host + '/api/custom_ml/cv_status',
                headers=json_headers(self.token),
                data=json.dumps({'id': task_id}),
            )
            if not resp.ok:
                raise
            resp_body = json.loads(resp.content)
            status = resp_body['payload']['status']
            if (status == 'ERROR'):
                print('ERROR')
                raise
            elif (status == 'FINISHED'):
                print('FINISHED 100%')
                return
            else:
                progress = resp_body['payload']['total_progress']
                if not silent:
                    print('{} {:.2f}%'.format(status, 100 * float(progress)), end='\r')
                time.sleep(1)
        print('')

    def __cv_results(self, task_id: int):
        resp = requests.post(
            self.host + '/api/custom_ml/cv_results',
            headers=json_headers(self.token),
            data=json.dumps({'id': task_id}),
        )
        if not resp.ok:
            raise
        return json.loads(resp.content)
