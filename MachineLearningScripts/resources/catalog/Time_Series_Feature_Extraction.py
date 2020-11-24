# -*- coding: utf-8 -*-
"""Proactive Summarize Data for Machine Learning

This module contains the Python script for the Summarize Data task.
"""
import ssl
import urllib.request

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

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
global compress_and_transfer_dataframe_in_variables
global assert_not_none_not_empty, is_not_none_not_empty
global get_input_variables, get_and_decompress_dataframe
global compute_global_model, get_summary, is_true

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
TIME_COLUMN = variables.get("TIME_COLUMN")
REF_COLUMN = variables.get("REF_COLUMN")
ALL_FEATURES = variables.get("ALL_FEATURES")

assert_not_none_not_empty(TIME_COLUMN, "TIME_COLUMN should be defined!")
assert_not_none_not_empty(REF_COLUMN, "REF_COLUMN should be defined!")
assert_not_none_not_empty(ALL_FEATURES, "ALL_FEATURES should be defined!")

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# Add the list of features that need to be extracted 
# Check the full list of features on this link:
# http://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
if is_true(ALL_FEATURES):
    extraction_settings = {
        "length": None,
        "absolute_sum_of_changes": None,
        "abs_energy": None,
        # "sample_entropy": None,
        "number_peaks": [{"n": 2}],
        "number_cwt_peaks": [{"n": 2}, {"n": 3}],
        "autocorrelation": [{"lag": 2}, {"lag": 3}]
        # "value_count": #"large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    }
# For convenience, three dictionaries are predefined and can be used right away
# ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
# MinimalFCParameters is used by default
else:
    extraction_settings = MinimalFCParameters()

extracted_features = extract_features(dataframe,
                                      column_id=REF_COLUMN,
                                      column_sort=TIME_COLUMN,
                                      default_fc_parameters=extraction_settings)
extracted_features[REF_COLUMN] = extracted_features.index

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe_in_variables(dataframe)
print("dataframe id (out): ", dataframe_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", input_variables['task.label_column'])

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)
