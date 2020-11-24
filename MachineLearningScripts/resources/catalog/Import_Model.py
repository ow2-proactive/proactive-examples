# -*- coding: utf-8 -*-
"""Proactive Import Model for Machine Learning

This module contains the Python script for the Import Model task.
"""
import ssl
import urllib.request
import bz2
import pickle
import sys
import uuid
import wget

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
global check_task_is_enabled, assert_not_none_not_empty

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

# Dumps the model
model_bin = pickle.dumps(model)
assert model_bin is not None

# Compress the model
compressed_model = bz2.compress(model_bin)

# Transfer model via Proactive variables
model_id = str(uuid.uuid4())
variables.put(model_id, compressed_model)

print("model id: ", model_id)
print('model size (original):   ', sys.getsizeof(model_bin), " bytes")
print('model size (compressed): ', sys.getsizeof(compressed_model), " bytes")

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.model_id", model_id)

# result = model_bin
# resultMetadata.put("file.extension", ".model")
# resultMetadata.put("file.name", "myModel.model")
# resultMetadata.put("content.type", "application/octet-stream")

print("END " + __file__)
