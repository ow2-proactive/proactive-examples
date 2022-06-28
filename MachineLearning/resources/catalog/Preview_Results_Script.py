"""Proactive Preview Results for Machine Learning

This module contains the Python script for the Preview Results task.
"""
import ssl
import urllib.request
import json

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
global check_task_is_enabled, assert_not_none_not_empty
global get_input_variables, get_and_decompress_dataframe
global preview_dataframe_in_task_result

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
    'task.label_column': None,
    'task.encode_map_json': None
}
get_input_variables(input_variables)

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if is_not_none_not_empty(LABEL_COLUMN):
    is_labeled_data = True
else:
    LABEL_COLUMN = input_variables['task.label_column']
    if is_not_none_not_empty(LABEL_COLUMN):
        is_labeled_data = True

encode_map_json = input_variables['task.encode_map_json']
encode_map = None
if encode_map_json is not None:
    encode_map = json.loads(encode_map_json)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# -------------------------------------------------------------
# Preview results
#
if encode_map is not None and is_labeled_data:
    # apply_encoder(dataframe, columns, encode_map, sep=",")
    encode_map['predictions'] = encode_map[LABEL_COLUMN]
    dataframe_aux = apply_encoder(dataframe, [LABEL_COLUMN, 'predictions'], encode_map)
    preview_dataframe_in_task_result(dataframe_aux)
else:
    preview_dataframe_in_task_result(dataframe, output_type=OUTPUT_TYPE)

# -------------------------------------------------------------
print("END " + __file__)
