#!/usr/bin/env python
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
from jadbio_internal.ml.tuning import cv_params

jad = ApiClient('https://exp.jadbio.com:4443', 'pkatsogr@gmail.com', '22222222')

# Initialize custom fs algorithms
ses = SESParams(max_k=2, threshold=0.001, max_vars=25)
lasso = LASSO(max_vars=50, penalty=1.0)

# Initialize custom models
dt = DT(mls=1, splits=1, alpha=0.01, criterion=SplitCriterion.Deviance, vars_f=SplitFunction.perc(50))
svm = SVM.polynomialSVM(degrees=3, cost=10, gamma=4)
logistic = Logistic(l=1.0)

## To perform fully automated analysis
#tuning = tuning_params.auto()
## To perform analysis with auto preprocessing and models but custom fs
#tuning = tuning_params.custom_fs(feature_selectors=[ses, lasso])
## To perform analysis with auto preprocessing and fs but custom models
#tuning = tuning_params.custom_models(models=[dt, logistic, svm])
# To perform a fully custom analysis
conf1 = HyperparamConf.with_default_pp(lasso, dt)
conf2 = HyperparamConf.with_default_pp(ses, svm)
conf3 = HyperparamConf.with_default_pp(ses, logistic)

tuning = tuning_params.custom([conf1, conf2, conf3])

# Run analysis
project = jad.project.find_project('standard')
d = jad.project.find_dataset(project, 'Alzheimer')
#
form = analysis_form.testing('Target', 'QUICK', 1, tuning).with_cv_params(cv_params.custom_repeats(1))
res = jad.ml.analyze_testing(dataset_id=d, form=form)

