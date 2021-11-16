# Copyright Activeeon 2007-2021. All rights reserved.

# -*- coding: utf-8 -*-
"""Proactive Split Data for Machine Learning

This module contains the Python script for the Split Data task.
"""
import ssl
import urllib.request

from sklearn.model_selection import train_test_split

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
global get_and_decompress_dataframe, compress_and_transfer_dataframe
global assert_not_none_not_empty, assert_valid_float
global assert_between, get_input_variables

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
TRAIN_SIZE = variables.get("TRAIN_SIZE")
assert_not_none_not_empty(TRAIN_SIZE, "TRAIN_SIZE should be defined!")
TRAIN_SIZE = assert_valid_float(TRAIN_SIZE, "TRAIN_SIZE should be a float!")
assert_between(TRAIN_SIZE, minvalue=0, maxvalue=1, msg="TRAIN_SIZE should be between 0 and 1!")
test_size = 1 - TRAIN_SIZE

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None,
    'task.feature_names': None,
    'task.encode_map_json': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# Split dataframe into train/test sets
X_train, X_test = train_test_split(dataframe, test_size=test_size)

dataframe1 = X_train.reset_index(drop=True)
dataframe2 = X_test.reset_index(drop=True)

dataframe_id1 = compress_and_transfer_dataframe(dataframe1)
dataframe_id2 = compress_and_transfer_dataframe(dataframe2)

print("dataframe id1 (out) [train set]: ", dataframe_id1)
print("dataframe id2 (out) [test set]:  ", dataframe_id2)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id_train", dataframe_id1)
resultMetadata.put("task.dataframe_id_test", dataframe_id2)
resultMetadata.put("task.label_column", input_variables['task.label_column'])
resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
resultMetadata.put("task.encode_map_json", input_variables['task.encode_map_json'])

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)