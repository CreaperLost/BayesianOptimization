#!/usr/bin/env python
from jadbio_internal.ml.algo.DT import DT
from jadbio_internal.ml.algo.Logistic import Logistic
from jadbio_internal.ml.algo.RF import RF
from jadbio_internal.ml.algo.SVM import SVM
from jadbio_internal.ml.algo.dt.SplitCriterion import SplitCriterion
from jadbio_internal.ml.algo.dt.SplitFunction import SplitFunction
from jadbio_internal.ml.algo.rf.Bagger import Bagger
from jadbio_internal.ml.algo.rf.LossFunction import LossFunction
from jadbio_internal.ml.analysis_type import AnalysisType
from jadbio_internal.ml.custom_cv_form import CVArgs
from jadbio_internal.ml.fs.LASSO import LASSO
from jadbio_internal.ml.fs.SESParams import SESParams

from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.hyperparam_conf import HyperparamConf

jad = ApiClient('https://dev2.jadbio.com:4443', 'pkatsogr', 'led13lemmy')

project = jad.project.find_project('standard')
d = jad.project.find_dataset(project, 'Alzheimer')
cv_args = CVArgs(dataset_id=d, target='Target', type=AnalysisType.CLASSIFICATION)

# to specify the number of threads(parallelism): cv_args.add_threads(12)
# to specify a group factor: cv_args.add_grouped_factor('var name')

# create a SES configuration
ses = SESParams(max_k=2, threshold=0.001, max_vars=25)
#
# # create a LASSO configuration
# lasso = LASSO(max_vars=50, penalty=1.0)
#
# # add a random forest configuration
dt = DT(mls=1, splits=1, alpha=0.01, criterion=SplitCriterion.Deviance, vars_f=SplitFunction.perc(50))
# rf = RF(nmodels=100, dt=dt, loss=LossFunction.Deviance, bagger=Bagger.Probability)
rf_conf = HyperparamConf.with_default_pp(ses, dt)
cv_args.add_conf('rf_conf', rf_conf)
res = jad.ml.cv(form=cv_args)

# # add a logistic regression conf
# logistic_conf = HyperparamConf.with_default_pp(ses, Logistic(l=1.0))
# cv_args.add_conf('logistic_conf', logistic_conf)
#
# # add a Polynomial SVM configuration
# svm_conf = HyperparamConf.with_default_pp(ses, SVM.polynomialSVM(degrees=3, cost=10, gamma=4))
# # same svm with LASSO as feature selector
# svm_conf_lasso = HyperparamConf.with_default_pp(lasso, SVM.polynomialSVM(degrees=3, cost=10, gamma=4))
# svm_conf2 = HyperparamConf.with_default_pp(lasso, SVM.polynomialSVM(degrees=2, cost=3, gamma=15))
# cv_args.add_conf('svm_conf', svm_conf)
# cv_args.add_conf('svm_conf_lasso', svm_conf_lasso)
# cv_args.add_conf('svm_conf2', svm_conf2)
#
#
# print(res.analysis_id)
#
# print(jad.apply_model(res.analysis_id, d))
# print(res.predictions.oos_preds_for_repeat('svm_conf', 0))
# print(res.predictions.oos_preds_for_repeat('svm_conf_lasso', 0))
#
# print(res.metrics.get('rf_conf'))
# print(res.metrics.get('svm_conf'))
# print(res.metrics.get('svm_conf_lasso'))
# print(res.metrics.get('svm_conf2'))
# print(res.metrics.get('logistic_conf'))
# res.target_info.valueForClass(0)
# print(res.runtime.conf_runtime_millis('logistic_conf'))
# print(res.best_model_preds.preds)