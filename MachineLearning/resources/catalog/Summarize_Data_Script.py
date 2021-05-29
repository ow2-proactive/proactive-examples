
# -*- coding: utf-8 -*-
"""Proactive Summarize Data for Machine Learning

This module contains the Python script for the Summarize Data task.
"""
import ssl
import urllib.request
import pandas as pd

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning/resources/Utils_Script/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global get_input_variables, get_and_decompress_dataframe
global compress_and_transfer_dataframe, compute_global_model, get_summary
global assert_not_none_not_empty, is_not_none_not_empty

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
REF_COLUMN = variables.get("REF_COLUMN")
assert_not_none_not_empty(REF_COLUMN, "REF_COLUMN should be defined!")
IGNORE_COLUMNS = [REF_COLUMN]

LABEL_COLUMN = variables.get("LABEL_COLUMN")
is_labelled_data = False
if is_not_none_not_empty(LABEL_COLUMN):
    IGNORE_COLUMNS.append(LABEL_COLUMN)
    is_labelled_data = True

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# Perform dataframe summarization
columns = dataframe.drop(IGNORE_COLUMNS, axis=1, inplace=False).columns.values
ncolumns = columns.shape[0]
bins = [10] * ncolumns
print(columns, ncolumns, bins)

model = None
GLOBAL_MODEL_TYPE = variables.get("GLOBAL_MODEL_TYPE")
if is_not_none_not_empty(GLOBAL_MODEL_TYPE):
    print('Computing the global model using ', GLOBAL_MODEL_TYPE)
    model = compute_global_model(dataframe, columns, bins, GLOBAL_MODEL_TYPE)
    print('Finished')

print('Summarizing data...')
data = get_summary(dataframe, columns, bins, model, GLOBAL_MODEL_TYPE, REF_COLUMN, LABEL_COLUMN)
print('Finished')

dataframe = pd.DataFrame.from_dict(data, orient='index')
cols_len = len(dataframe.columns)
dataframe.columns = list(range(0, cols_len))

COLUMNS_NAME = {0: REF_COLUMN}
if is_labelled_data:
    COLUMNS_NAME = {0: REF_COLUMN, cols_len - 1: LABEL_COLUMN}
dataframe.rename(index=str, columns=COLUMNS_NAME, inplace=True)

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe(dataframe)
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