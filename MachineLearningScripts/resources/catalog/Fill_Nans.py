# -*- coding: utf-8 -*-
"""Proactive Fill Nans for Machine Learning

This module contains the Python script for the Fill Nans task.
"""
import json
import urllib.request
import numpy as np

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global compress_and_transfer_dataframe_in_variables
global assert_not_none_not_empty, get_input_variables
global get_and_decompress_dataframe

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# Replace `inf` and `-inf` with `nan`
dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace `nan` by `zeros` or by using a `fill map`
FILL_MAP = variables.get("FILL_MAP")
if FILL_MAP is not None and FILL_MAP is not "":
    fill_map = json.loads(FILL_MAP)
    dataframe.fillna(value=fill_map, inplace=True)
else:
    dataframe.fillna(0, inplace=True)

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
