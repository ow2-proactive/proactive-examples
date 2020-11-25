# -*- coding: utf-8 -*-
"""Proactive Export Data for Machine Learning

This module contains the Python script for the Export Data task.
"""
import ssl
import urllib.request

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
global check_task_is_enabled, assert_not_none_not_empty
global get_input_variables, get_and_decompress_dataframe
global export_dataframe_in_task_result

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
OUTPUT_TYPE = variables.get("OUTPUT_TYPE")
assert_not_none_not_empty(OUTPUT_TYPE, "OUTPUT_TYPE should be defined!")

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# -------------------------------------------------------------
# Preview results
#
export_dataframe_in_task_result(dataframe, output_type=OUTPUT_TYPE)

# -------------------------------------------------------------
print("END " + __file__)
