import unittest
import pathlib
import sys
import time
import os
from jadbio_internal.ml import analysis_form
from jadbio_internal.ml.algo.DT import DT
from jadbio_internal.ml.algo.Logistic import Logistic
from jadbio_internal.ml.algo.SVM import SVM
from jadbio_internal.ml.algo.dt.SplitCriterion import SplitCriterion
from jadbio_internal.ml.algo.dt.SplitFunction import SplitFunction

from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.fs.LASSO import LASSO
from jadbio_internal.ml.fs.SESParams import SESParams
from jadbio_internal.ml.hyperparam_conf import HyperparamConf
from jadbio_internal.ml.tuning import tuning_params

class Tests(unittest.TestCase):
    TIMEOUT = 2 * 10 * 60
    client = None
    pid = None

    @classmethod
    def setUpClass(cls):
        host = os.environ['JADBIO_API_URL']
        user = os.environ['JADBIO_USER']
        password = os.environ['JADBIO_PASSWORD']
        cur_folder_pth = str(pathlib.Path(__file__).parent.absolute())
        cls.pth_to_resources = cur_folder_pth+'/'
        cls.client = ApiClient(host, user, password)
        cls.pid = cls.client.create_project('custom_analysis')

    @classmethod
    def tearDownClass(cls):
        if cls.pid is not None:
            cls.client.project.delete_project(cls.pid)

    def test_run_testing_analysis(self):
        project = self.client.load_project('custom_analysis')
        dataset = self.client.upload_dataset_if_not_exists(project, 'Alzheimer', self.pth_to_resources+'datasets/alzheimer.csv')
        ses = SESParams(max_k=2, threshold=0.001, max_vars=25)
        lasso = LASSO(max_vars=50, penalty=1.0)
        dt = DT(mls=1, splits=1, alpha=0.01, criterion=SplitCriterion.Deviance, vars_f=SplitFunction.perc(50))
        svm = SVM.polynomialSVM(degrees=3, cost=10, gamma=4)
        logistic = Logistic(l=1.0)
        conf1 = HyperparamConf.with_default_pp(lasso, dt)
        conf2 = HyperparamConf.with_default_pp(ses, svm)
        conf3 = HyperparamConf.with_default_pp(ses, logistic)
        tuning = tuning_params.custom([conf1, conf2, conf3])
        form = analysis_form.testing('Target', 'QUICK', 1, tuning)
        res = self.client.ml.analyze_testing(dataset_id=dataset, form=form)
        self.assertIn('oosPredictions', res)
        self.assertIn('bestModel', res)
        self.assertIn('optMetric', res)
        self.assertIn('splitIndices', res)

