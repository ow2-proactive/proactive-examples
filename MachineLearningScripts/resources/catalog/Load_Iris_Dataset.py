# -*- coding: utf-8 -*-
"""Proactive Load Iris Dataset for Machine Learning

This module contains the Python script for the Load Iris Dataset task.
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
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global compress_and_transfer_dataframe_in_variables
global assert_not_none_not_empty

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
IMPORT_FROM = variables.get("IMPORT_FROM")
FILE_PATH = variables.get("FILE_PATH")
FILE_DELIMITER = variables.get("FILE_DELIMITER")
LABEL_COLUMN = variables.get("LABEL_COLUMN")

assert_not_none_not_empty(IMPORT_FROM, "IMPORT_FROM should be defined!")
assert_not_none_not_empty(FILE_PATH, "FILE_PATH should be defined!")
assert_not_none_not_empty(FILE_DELIMITER, "FILE_DELIMITER should be defined!")

# -------------------------------------------------------------
# Load file
#
if IMPORT_FROM.upper() == "PA:USER_FILE":
    print("Importing file from the user space")
    userspaceapi.connect()
    out_file = gateway.jvm.java.io.File(FILE_PATH)
    userspaceapi.pullFile(FILE_PATH, out_file)

if IMPORT_FROM.upper() == "PA:GLOBAL_FILE":
    print("Importing file from the global space")
    globalspaceapi.connect()
    out_file = gateway.jvm.java.io.File(FILE_PATH)
    globalspaceapi.pullFile(FILE_PATH, out_file)

dataframe = pd.read_csv(FILE_PATH, FILE_DELIMITER)

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe_in_variables(dataframe)
print("dataframe id (out): ", dataframe_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", LABEL_COLUMN)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)
