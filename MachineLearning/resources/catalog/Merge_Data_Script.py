
# -*- coding: utf-8 -*-
"""Proactive Merge Data for Machine Learning

This module contains the Python script for the Merge Data task.
"""
import ssl
import urllib.request
import pandas as pd

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Get schedulerapi access and acquire session id
schedulerapi.connect()
sessionid = schedulerapi.getSession()

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/ai-machine-learning/resources/Utils_Script/raw"
req = urllib.request.Request(PA_PYTHON_UTILS_URL)
req.add_header('sessionid', sessionid)
if PA_PYTHON_UTILS_URL.startswith('https'):
    content = urllib.request.urlopen(req, context=ssl._create_unverified_context()).read()
else:
    content = urllib.request.urlopen(req).read()
exec(content, globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global get_and_decompress_dataframe, compress_and_transfer_dataframe
global get_input_variables, get_input_variables_from_key
global assert_not_none_not_empty

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
REF_COLUMN = variables.get("REF_COLUMN")
assert_not_none_not_empty(REF_COLUMN, "REF_COLUMN should be defined!")

input_variables = {
    'task.label_column': None
}
get_input_variables(input_variables)

input_dataframes = {
    'dataframe_id1': None,
    'dataframe_id2': None
}
get_input_variables_from_key(input_dataframes, key='task.dataframe_id')
dataframe_id1 = input_dataframes['dataframe_id1']
dataframe_id2 = input_dataframes['dataframe_id2']

assert_not_none_not_empty(dataframe_id1, __file__ + " need two dataframes!")
assert_not_none_not_empty(dataframe_id2, __file__ + " need two dataframes!")

print("dataframe id1 (in): ", dataframe_id1)
print("dataframe id2 (in): ", dataframe_id2)

dataframe1 = get_and_decompress_dataframe(dataframe_id1)
dataframe2 = get_and_decompress_dataframe(dataframe_id2)

dataframe = pd.merge(dataframe1, dataframe2, on=[REF_COLUMN], how='outer')

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