
# -*- coding: utf-8 -*-
"""Proactive Import Model for Machine Learning

This module contains the Python script for the Import Model task.
"""
import ssl
import urllib.request
import pickle
import wget

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
global check_task_is_enabled, assert_not_none_not_empty, compress_and_transfer_model

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
MODEL_URL = variables.get("MODEL_URL")
assert_not_none_not_empty(MODEL_URL, "MODEL_URL should be defined!")

# Download model from URL
filename = wget.download(str(MODEL_URL))

# Load model in memory
model = pickle.load(open(filename, "rb"))

# Transfer model for the next tasks
model_id = compress_and_transfer_model(model)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.model_id", model_id)

print("END " + __file__)