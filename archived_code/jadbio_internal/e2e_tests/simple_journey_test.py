import unittest
import pathlib
import os
from jadbio_internal.ml import analysis_form
from jadbio_internal.api_client import ApiClient

class Tests(unittest.TestCase):
    TIMEOUT = 2 * 10 * 60
    client = None
    pid = None

    @classmethod
    def setUpClass(cls):
        host = os.environ['JADBIO_HOST']
        user = os.environ['JADBIO_USER']
        password = os.environ['JADBIO_PASSWORD']
        cur_folder_pth = str(pathlib.Path(__file__).parent.absolute())
        cls.pth_to_resources = cur_folder_pth+'/'
        cls.client = ApiClient(host, user, password)
        cls.pid = cls.client.create_project('simple_journey')

    @classmethod
    def tearDownClass(cls):
        if cls.pid is not None:
            cls.client.project.delete_project(cls.pid)

    def test_classification(self):
        project = self.client.load_project('simple_journey')
        dataset = self.client.upload_dataset_if_not_exists(project, 'Alzheimer', self.pth_to_resources+'datasets/alzheimer.csv')
        form = analysis_form.classification('Target', 'QUICK', 1)
        res = self.client.ml.analyze(dataset_id=dataset, form=form)
        self.assertIn('models', res)
        self.assertIn('overview', res)
        self.assertIn('jad_version', res)
        print(dataset)
