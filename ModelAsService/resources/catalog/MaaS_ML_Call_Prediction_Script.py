import json
import requests
import urllib.request
import ssl
import pandas as pd

global variables

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning/resources/Utils_Script/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global get_input_variables, get_and_decompress_dataframe, preview_dataframe_in_task_result
global raiser_ex

# -------------------------------------------------------------
# Get data from the propagated variables
#
SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get(
    "SERVICE_TOKEN_PROPAGATED")
assert SERVICE_TOKEN is not None

API_PREDICT = variables.get("PREDICT_EXTENSION") if variables.get("PREDICT_EXTENSION") else raiser_ex(
    "PREDICT_EXTENSION is None")
API_ENDPOINT = variables.get("PREDICT_MODEL_ENDPOINT") if variables.get("PREDICT_MODEL_ENDPOINT") else variables.get(
    "ENDPOINT_MODEL")
assert API_ENDPOINT is not None

API_PREDICT_ENDPOINT = API_ENDPOINT + API_PREDICT
print("API_PREDICT_ENDPOINT: ", API_PREDICT_ENDPOINT)

INPUT_DATA = variables.get("INPUT_DATA")
DATA_DRIFT_DETECTOR = variables.get("DATA_DRIFT_DETECTOR")

input_variables = {
    'task.dataframe_id_test': None,
    'task.dataframe_id': None,
    'task.label_column': None,
    'task.feature_names': None,
    'task.dataframe_sampled_id': None
}
get_input_variables(input_variables)

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
    is_labeled_data = True
else:
    LABEL_COLUMN = input_variables['task.label_column']
    if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
        is_labeled_data = True

dataframe_sampled_id = input_variables['task.dataframe_sampled_id']
dataframe_sampled = get_and_decompress_dataframe(dataframe_sampled_id) 
print("dataframe_sampled:\n", dataframe_sampled.head())
dataframe_sampled_json = dataframe_sampled.to_json(orient='values') 

dataframe_columns_name = None
if input_variables['task.dataframe_id_test'] is not None:
    dataframe_id = input_variables['task.dataframe_id_test']
    dataframe = get_and_decompress_dataframe(dataframe_id)
    if is_labeled_data:
        dataframe_test = dataframe.drop([LABEL_COLUMN], axis=1, inplace=False)
    else:
        dataframe_test = dataframe.copy()
    dataframe_columns_name = dataframe_test.columns.values
    dataframe_json = dataframe_test.to_json(orient='values')
elif input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
    dataframe = get_and_decompress_dataframe(dataframe_id)
    if is_labeled_data:
        dataframe_test = dataframe.drop([LABEL_COLUMN], axis=1, inplace=False)
    else:
        dataframe_test = dataframe.copy()
    dataframe_columns_name = dataframe_test.columns.values
    dataframe_json = dataframe_test.to_json(orient='values')
elif INPUT_DATA is not None and INPUT_DATA is not "":
    dataframe_json = variables.get("INPUT_DATA")
else:
    raiser_ex("There is no input data")

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
data = {'predict_dataframe_json': dataframe_json, 'api_token': SERVICE_TOKEN, 'detector': DATA_DRIFT_DETECTOR}
data_json = json.dumps(data)
req = requests.post(API_PREDICT_ENDPOINT, data=data_json, headers=headers, verify=False)

# predictions = json.loads(req.text)
# print("predictions:\n", predictions)
response = json.loads(req.text)
predict_and_drifts = json.loads(response)
predictions = predict_and_drifts["predictions"]
print("predictions:\n", predictions)
drifts = predict_and_drifts["drifts"]
print("drifts:\n", drifts)

# predictions = pd.read_json(predictions, orient='records')
df_dataframe = pd.read_json(dataframe_json, orient='records')
if dataframe_columns_name is not None:
    df_dataframe.columns = list(dataframe_columns_name)
dataframe = df_dataframe.assign(predictions=predictions)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)
