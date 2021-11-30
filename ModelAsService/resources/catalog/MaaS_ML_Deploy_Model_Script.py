import os
import sys
import wget
import requests
import urllib.request
import ssl
from urllib.parse import unquote

global variables, resultMetadata

# -------------------------------------------------------------
# Get schedulerapi access and acquire session id
schedulerapi.connect()
sessionid = schedulerapi.getSession()

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning/resources/Utils_Script/raw"
req = urllib.request.Request(PA_PYTHON_UTILS_URL)
req.add_header('sessionid', sessionid)
if PA_PYTHON_UTILS_URL.startswith('https'):
    content = urllib.request.urlopen(req, context=ssl._create_unverified_context()).read()
else:
    content = urllib.request.urlopen(req).read()
exec(content, globals())
global get_input_variables, compress_and_transfer_dataframe
global get_and_decompress_dataframe, get_and_decompress_json_dataframe
global get_and_decompress_model, save_model, raiser_ex

# -------------------------------------------------------------
# Get variables
#
DRIFT_DETECTION_WINDOW_SIZE = variables.get("DRIFT_DETECTION_WINDOW_SIZE")
LABEL_COLUMN = variables.get("LABEL_COLUMN")
API_DEPLOY = variables.get("API_EXTENSION") if variables.get("API_EXTENSION") else raiser_ex("API_EXTENSION is None")
LOG_PREDICTIONS = variables.get("LOG_PREDICTIONS") if variables.get("LOG_PREDICTIONS") else raiser_ex("LOG_PREDICTIONS is None")
SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get(
    "SERVICE_TOKEN_PROPAGATED")
API_ENDPOINT = variables.get("DEPLOY_MODEL_ENDPOINT") if variables.get("DEPLOY_MODEL_ENDPOINT") else variables.get(
    "ENDPOINT_MODEL")
MODEL_NAME = variables.get("MODEL_NAME") if variables.get("MODEL_NAME") else variables.get(
    "MODEL_NAME")
MODEL_VERSION = variables.get("MODEL_VERSION") if variables.get("MODEL_VERSION") else variables.get(
    "MODEL_VERSION")
BASELINE_DATA_URL = variables.get("BASELINE_DATA_URL")

API_DEPLOY_ENDPOINT = API_ENDPOINT + API_DEPLOY
print("API_DEPLOY_ENDPOINT: ", API_DEPLOY_ENDPOINT)

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.model_id': None,
    'task.feature_names': None,
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

model_id = input_variables['task.model_id']
model_path = os.path.join(os.getcwd(), "model.pkl")
if model_id is not None and variables.get(model_id) is not None:
    model = get_and_decompress_model(model_id)
    save_model(model, model_path)
else:
    MODEL_URL = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser_ex("MODEL_URL is None")
    wget.download(MODEL_URL, model_path)
print('model size (original):   ', sys.getsizeof(model_path), " bytes")
model_file = open(model_path, 'rb')

dataframe_id = None
if (input_variables['task.dataframe_id'] is not None) and (int(DRIFT_DETECTION_WINDOW_SIZE)!=0):
    dataframe_id = input_variables['task.dataframe_id']
    dataframe = get_and_decompress_dataframe(dataframe_id)
    dataframe_sampled = dataframe.sample(n=int(DRIFT_DETECTION_WINDOW_SIZE))
    #dataframe_sampled_id = compress_and_transfer_dataframe(dataframe_sampled)
    columns = list(dataframe.columns)
    columns_copy = columns.copy()
    columns_copy.remove(input_variables['task.label_column'])
    baseline_dataframe = dataframe_sampled[columns_copy]
    baseline_data_path = os.path.join(os.getcwd(), "baseline_data.csv")
    baseline_dataframe.to_csv(baseline_data_path, index=False)
    baseline_data = open(baseline_data_path, 'rb')
    files = {'model_file': model_file, 'baseline_data': baseline_data}

elif BASELINE_DATA_URL:
    baseline_data_url = unquote(BASELINE_DATA_URL)
    baseline_data_path = os.path.join(os.getcwd(), "baseline_data.csv")
    wget.download(baseline_data_url, baseline_data_path)
    print("[INFO] baseline_data donwloaded")
    baseline_data = open(baseline_data_path, 'rb')
    files = {'model_file': model_file, 'baseline_data': baseline_data}

else:    
    files = {'model_file': model_file}

data = {'api_token': SERVICE_TOKEN, 'log_predictions': LOG_PREDICTIONS, 'model_name': MODEL_NAME, 'model_version': MODEL_VERSION}

resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
#resultMetadata.put("task.dataframe_sampled_id", dataframe_sampled_id)

try:
    req = requests.post(API_DEPLOY_ENDPOINT, files=files, data=data, verify=False)
    print(req.text)
finally:
    model_file.close()