# -*- coding: utf-8 -*-
"""Proactive Feature_Vector_Extractor for Machine Learning

This module contains the Python script for the Feature_Vector_Extractor task.
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
global check_task_is_enabled, get_input_variables
global get_and_decompress_dataframe, compress_and_transfer_dataframe
global preview_dataframe_in_task_result


# Useful when there is multiple identifiers in a single row
def id_extraction(session_col=None):
    return str(session_col).split(' ')


# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
SESSION_COLUMN = variables.get("SESSION_COLUMN")
DATAFRAME_JSON = variables.get("DATAFRAME_JSON")
FILE_OUT_FEATURES = variables.get("FILE_OUT_FEATURES")
PATTERN_COLUMN = variables.get("PATTERN_COLUMN")
PATTERNS_COUNT_FEATURES = variables.get("PATTERNS_COUNT_FEATURES")
STATE_COUNT_FEATURES_VARIABLES = variables.get("STATE_COUNT_FEATURES_VARIABLES")

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

# -------------------------------------------------------------
# Extract variables
#
STATE_VARIABLES_INTER = variables.get("STATE_VARIABLES")
COUNT_VARIABLES_INTER = variables.get("COUNT_VARIABLES")
STATE_VARIABLES = STATE_VARIABLES_INTER.split(",")
COUNT_VARIABLES = COUNT_VARIABLES_INTER.split(",")
print("State variables:\n", STATE_VARIABLES)
print("Count variables:\n", COUNT_VARIABLES)

df_pattern_features = pd.DataFrame.empty
df_state_features = pd.DataFrame.empty
df_count_features = pd.DataFrame.empty

df_structured_logs = dataframe
pattern_number = int(df_structured_logs[PATTERN_COLUMN].max())

feature_vector = []
dict_block_features = {}
variables_name = list(df_structured_logs)
state_features_names = []
dict_states = {}
# dict_variables_set = {}
dict_variables_blk = {}
dict_block_features_state = {}
dict_block_features_state_1 = {}
dict_variables_set = {}

# -------------------------------------------------------------
# Extract the state variables
#
for i in range(len(STATE_VARIABLES)):
    variables_count = df_structured_logs[STATE_VARIABLES[i]].value_counts()
    for j in range(len(variables_count.keys())):
        dict_states[STATE_VARIABLES[i]] = variables_count.keys()
        state_features_names.append(variables_count.keys()[j])


for index, row in df_structured_logs.iterrows():
    if not (row[SESSION_COLUMN] is None):
        ids_list = id_extraction(row[SESSION_COLUMN])
        # -------------------------------------------------------------
        # Features (count pattern)
        #
        if PATTERNS_COUNT_FEATURES == 'True':
            j = int(row[PATTERN_COLUMN] - 1)
            for i in range(len(ids_list)):
                # dict_variables_blk = {}
                # update existing entry
                if ids_list[i] in dict_block_features:
                    features = dict_block_features.get(ids_list[i])
                    features[j] = features[j] + 1
                    dict_block_features[ids_list[i]] = features
                # add new entry
                else:
                    feature_vector = [0] * pattern_number
                    feature_vector[j] = feature_vector[j] + 1
                    dict_block_features[ids_list[i]] = feature_vector
        # -------------------------------------------------------------
        # Features (count state + variables)
        #
        if STATE_COUNT_FEATURES_VARIABLES == 'True':
            for f in range(len(ids_list)):
                # Update existing entry
                if ids_list[f] in dict_block_features_state_1:
                    features_count = dict_block_features_state_1.get(ids_list[f])
                    m = 0
                    for i in range(len(STATE_VARIABLES)):
                        for j in range(len(dict_states[STATE_VARIABLES[i]])):
                            if row[STATE_VARIABLES[i]] == dict_states[STATE_VARIABLES[i]][j]:
                                features_count[m] = features_count[m] + 1
                                dict_block_features_state[dict_states[STATE_VARIABLES[i]][j]] = features_count
                                dict_block_features_state_1[ids_list[f]] = features_count
                            m = m + 1

                    for h in range(len(COUNT_VARIABLES)):
                        table_of_variable = dict_variables_blk[ids_list[f]].get(COUNT_VARIABLES[h])
                        if (str(row[COUNT_VARIABLES[h]]) not in table_of_variable) and not (
                                row[COUNT_VARIABLES[h]] is None):
                            dict_variables_blk[ids_list[f]][COUNT_VARIABLES[h]].append(str(row[COUNT_VARIABLES[h]]))
                            features_count[m] = features_count[m] + 1
                            dict_block_features_state_1[ids_list[f]] = features_count
                        m = m + 1
                # Add new entry
                else:
                    feature_vector_state_variables = [0] * (len(state_features_names) + len(COUNT_VARIABLES))
                    m = 0
                    for i in range(len(STATE_VARIABLES)):
                        for j in range(len(dict_states[STATE_VARIABLES[i]])):
                            if row[STATE_VARIABLES[i]] == dict_states[STATE_VARIABLES[i]][j]:
                                feature_vector_state_variables[m] = feature_vector_state_variables[m] + 1
                                dict_block_features_state[
                                    dict_states[STATE_VARIABLES[i]][j]] = feature_vector_state_variables
                                dict_block_features_state_1[ids_list[f]] = feature_vector_state_variables
                            m = m + 1
                    dict_variables_set_1 = {}
                    for h in range(len(COUNT_VARIABLES)):
                        dict_variables_set_1[COUNT_VARIABLES[h]] = []
                        if not (row[COUNT_VARIABLES[h]] is None):
                            dict_variables_set_1[COUNT_VARIABLES[h]].append(str(row[COUNT_VARIABLES[h]]))
                            feature_vector_state_variables[m] = feature_vector_state_variables[m] + 1
                        m = m + 1
                        dict_block_features_state_1[ids_list[f]] = feature_vector_state_variables
                        dict_variables_blk[ids_list[f]] = dict_variables_set_1
# -------------------------------------------------------------
# Save the different features in a dataframe
#
frames = []
if PATTERNS_COUNT_FEATURES == 'True':
    features = dict_block_features.values()
    block_ids = dict_block_features.keys()
    df_pattern_features = pd.DataFrame(dict_block_features,
                                       index=["pattern " + str(i) for i in range(1, pattern_number + 1)]).T
    frames.append(df_pattern_features)

if STATE_COUNT_FEATURES_VARIABLES == 'True':
    df_state_features = pd.DataFrame(dict_block_features_state_1, index=state_features_names + COUNT_VARIABLES).T
    frames.append(df_state_features)

if not frames:
    df_features = pd.DataFrame.empty
    print("ERROR: No features extracted, check your input variables")
else:
    df_features = pd.concat(frames, axis=1)

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe = df_features
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
# Save the linked variables
columns_name = df_features.columns
variables.put("COLUMNS_NAME_JSON", pd.Series(columns_name).to_json())

print("END " + __file__)
