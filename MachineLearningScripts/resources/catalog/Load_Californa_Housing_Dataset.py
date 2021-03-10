# -*- coding: utf-8 -*-
"""Proactive Load_Californa_Housing_Dataset for Machine Learning

This module contains the Python script for the Load_Californa_Housing_Dataset task.
"""
import ssl
import urllib.request
import sys, bz2, uuid
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global compress_and_transfer_dataframe

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
california_housing = fetch_california_housing()
dataframe_load = pd.DataFrame(california_housing.data)
dataframe_load.columns = california_housing.feature_names 
data_label = california_housing.target
label_column = "LABEL"
dataframe = dataframe_load.assign(LABEL=data_label)
dataframe, neglected = train_test_split(dataframe, test_size=0.98)
feature_names = dataframe.columns

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe(dataframe)
print("dataframe id (out): ", dataframe_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", label_column)
resultMetadata.put("task.feature_names", feature_names)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)