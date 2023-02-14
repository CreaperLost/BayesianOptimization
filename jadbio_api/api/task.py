import json
import time

import requests
import sys
sys.path.insert(1, '..')
from .auth import json_headers
from .job import Job


def __get_task_state(host:str, token: str, task_id: str):
    return requests.post(
        host + '/api/dataArray/getTaskState',
        headers = json_headers(token),
        data=json.dumps({'id': task_id}),
        verify=True
    )

def wait_for_job(host: str, token: str, job: Job):
    while True:
        resp = __get_task_state(host, token, job.id)
        if not resp.ok:
            print(resp.content)
            raise
        resp_body = json.loads(resp.content)
        status = resp_body['state']
        if(status=='ERROR'):
            print('ERROR')
            raise
        elif(status=='FINISHED'):
            print('FINISHED 100%')
            return resp_body['did']
        else:
            print('RUNNING')
            time.sleep(1)