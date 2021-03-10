import os
import requests
import sys
import wget
import urllib.request

global variables, resultMetadata

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global get_input_variables, compress_and_transfer_dataframe
global get_and_decompress_dataframe, get_and_decompress_json_dataframe
global get_and_decompress_model, save_model, raiser_ex

# -------------------------------------------------------------
# Get variables
#
DRIFT_DETECTION_WINDOW_SIZE = variables.get("DRIFT_DETECTION_WINDOW_SIZE")
LABEL_COLUMN = variables.get("LABEL_COLUMN")
API_DEPLOY = variables.get("API_EXTENSION") if variables.get("API_EXTENSION") else raiser_ex("API_EXTENSION is None")
SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get(
    "SERVICE_TOKEN_PROPAGATED")
API_ENDPOINT = variables.get("DEPLOY_MODEL_ENDPOINT") if variables.get("DEPLOY_MODEL_ENDPOINT") else variables.get(
    "ENDPOINT_MODEL")
API_DEPLOY_ENDPOINT = API_ENDPOINT + API_DEPLOY
print("API_DEPLOY_ENDPOINT: ", API_DEPLOY_ENDPOINT)

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.model_id': None,
    'task.model_metadata_id': None,
    'task.feature_names': None,
    'task.dataframe_id': None,
    'task.label_column': None
}
get_input_variables(input_variables)

model_id = input_variables['task.model_id']
model_metadata_id = input_variables['task.model_metadata_id']

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']

dataframe = get_and_decompress_dataframe(dataframe_id)

dataframe_sampled = dataframe.sample(n=int(DRIFT_DETECTION_WINDOW_SIZE))
columns = list(dataframe.columns)
columns_copy = columns.copy()
columns_copy.remove(input_variables['task.label_column'])
ds = dataframe_sampled[columns_copy]
dataframe_sampled_id = compress_and_transfer_dataframe(dataframe_sampled)

model_path = os.path.join(os.getcwd(), "model.pkl")
if model_id is not None and variables.get(model_id) is not None:
    model = get_and_decompress_model(model_id)
    save_model(model, model_path)
else:
    MODEL_URL = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser_ex("MODEL_URL is None")
    wget.download(MODEL_URL, model_path)
print('model size (original):   ', sys.getsizeof(model_path), " bytes")

sampled_data_path = os.path.join(os.getcwd(), "baseline_data.csv")
ds.to_csv(sampled_data_path)
model_file = open(model_path, 'rb')
baseline_data = open(sampled_data_path, 'rb')
files = {'model_file': model_file, 'baseline_data': baseline_data}
data = {'api_token': SERVICE_TOKEN}

# [deprecated]
# import warnings
# warnings.warn("model_metadata is deprecated", DeprecationWarning)
# if model_metadata_id is not None and variables.get(model_metadata_id) is not None:
#     print("model_metadata_id: ", model_metadata_id)
#     model_metadata_json = get_and_decompress_json_dataframe(model_metadata_id)
#     print("model_metadata_json: ", model_metadata_json)
#     data['model_metadata_json'] = model_metadata_json

resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
resultMetadata.put("task.dataframe_sampled_id", dataframe_sampled_id)

try:
    req = requests.post(API_DEPLOY_ENDPOINT, files=files, data=data, verify=False)
    print(req.text)
finally:
    model_file.close()
