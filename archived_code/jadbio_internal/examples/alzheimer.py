#!/usr/bin/env python
from jadbio_internal.ml import analysis_form
from jadbio_internal.ml.algo.DT import DT
from jadbio_internal.ml.algo.Logistic import Logistic
from jadbio_internal.ml.algo.RF import RF
from jadbio_internal.ml.algo.SVM import SVM
from jadbio_internal.ml.algo.dt.SplitCriterion import SplitCriterion
from jadbio_internal.ml.algo.dt.SplitFunction import SplitFunction
from jadbio_internal.ml.algo.rf.Bagger import Bagger
from jadbio_internal.ml.algo.rf.LossFunction import LossFunction
from jadbio_internal.ml.analysis_form import AnalysisForm
from jadbio_internal.ml.analysis_type import AnalysisType
from jadbio_internal.ml.custom_cv_form import CVArgs
from jadbio_internal.ml.fs.LASSO import LASSO
from jadbio_internal.ml.fs.SESParams import SESParams

from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.hyperparam_conf import HyperparamConf
import os

user = os.environ['JADBIO_USER']
creds = os.environ['JADBIO_PASSWORD']
host = os.environ['JADBIO_API_URL']
jad = ApiClient(host, user, creds)
project = jad.project.find_project('standard')
d = jad.project.find_dataset(project, 'Alzheimer')
form = analysis_form.classification('Target', 'EXTENSIVE', 6)
res = jad.ml.submit_analysis(dataset_id=d, form=form)
