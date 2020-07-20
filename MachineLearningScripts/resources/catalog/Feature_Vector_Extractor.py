__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2
import sys
import uuid

import pandas as pd


# Useful when there is multiple identifiers in a single row
def id_extraction(session_col=None):
    session_col = str(session_col)
    ids_list = session_col.split(' ')
    return ids_list


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
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe_json = variables.get(dataframe_id)
assert dataframe_json is not None

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

DATAFRAME_JSON = bz2.decompress(dataframe_json).decode()
df_structured_logs = pd.read_json(DATAFRAME_JSON, orient='split')
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

dataframe_json = df_features.to_json(orient='split').encode()
compressed_data = bz2.compress(dataframe_json)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

print("dataframe id (out): ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(dataframe_json), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", input_variables['task.label_column'])

# -------------------------------------------------------------
# Preview results
#
LIMIT_OUTPUT_VIEW = variables.get("LIMIT_OUTPUT_VIEW")
LIMIT_OUTPUT_VIEW = 5 if LIMIT_OUTPUT_VIEW is None else int(LIMIT_OUTPUT_VIEW)
if LIMIT_OUTPUT_VIEW > 0:
    print("task result limited to: ", LIMIT_OUTPUT_VIEW, " rows")
    dataframe = df_features.head(LIMIT_OUTPUT_VIEW).copy()
result = ''
with pd.option_context('display.max_colwidth', -1):
    result = dataframe.to_html(escape=False, classes='table table-bordered table-striped', justify='center')
result = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Machine Learning Preview</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" 
integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>
<body class="container">
<h1 class="text-center my-4" style="color:#003050;">Data Preview</h1>
<div style="text-align:center">{0}</div>
</body></html>""".format(result)
result = result.encode('utf-8')
resultMetadata.put("file.extension", ".html")
resultMetadata.put("file.name", "output.html")
resultMetadata.put("content.type", "text/html")

# -------------------------------------------------------------
# Save the linked variables
columns_name = df_features.columns
variables.put("COLUMNS_NAME_JSON", pd.Series(columns_name).to_json())

print("END " + __file__)
