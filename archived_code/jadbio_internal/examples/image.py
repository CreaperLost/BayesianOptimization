#!/usr/bin/env python
import pandas
from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.analysis_form import AnalysisForm

client = ApiClient("https://api.jadbio.com:443", "pkatsogr", "12345678")

# loc = '/home/pkatsogr/datasets/images/histo/'
# loc = '/home/pkatsogr/datasets/images/xrays3_train/'

project = client.project.find_project('images')

# d = client.image.upload_type2(project, 'xrays3_train', loc)

d = client.project.find_dataset(project, 'histo1_train')

a = client.ml.submit_analysis(d, AnalysisForm.classification('target', 'QUICK', 6))

# v = client.ml.submit_validation(d, 13589)
