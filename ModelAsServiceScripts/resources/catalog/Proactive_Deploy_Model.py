import bz2
import os
import requests
import sys
import wget
import pandas as pd
import urllib.request

PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global compress_and_transfer_dataframe_in_variables
global assert_not_none_not_empty, assert_valid_float
global assert_between, get_input_variables
global get_and_decompress_dataframe

def raiser(msg): raise Exception(msg)


MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
SAMPLED_DATA_PATH = os.path.join(os.getcwd(), "baseline_data.csv")

# Get variables
DRIFT_DETECTION_WINDOW_SIZE = variables.get("DRIFT_DETECTION_WINDOW_SIZE")
LABEL_COLUMN = variables.get("LABEL_COLUMN")
API_DEPLOY = variables.get("API_EXTENSION") if variables.get("API_EXTENSION") else raiser("API_EXTENSION is None")
SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get(
    "SERVICE_TOKEN_PROPAGATED")
API_ENDPOINT = variables.get("DEPLOY_MODEL_ENDPOINT") if variables.get("DEPLOY_MODEL_ENDPOINT") else variables.get(
    "ENDPOINT_MODEL")
API_DEPLOY_ENDPOINT = API_ENDPOINT + API_DEPLOY
print("API_DEPLOY_ENDPOINT: ", API_DEPLOY_ENDPOINT)

# Get model
input_variables = {
    'task.model_id': None,
    'task.model_metadata_id': None,
    'task.feature_names': None,
    'task.dataframe_id': None,
    'task.label_column': None
}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

model_id = input_variables['task.model_id']
model_metadata_id = input_variables['task.model_metadata_id']

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']

dataframe_json = get_and_decompress_json_dataframe(dataframe_id)
dataframe = get_and_decompress_dataframe(dataframe_id)

dataframe_sampled = dataframe.sample (n=int(DRIFT_DETECTION_WINDOW_SIZE))
columns=list(dataframe.columns)
columns_copy=columns.copy()
columns_copy.remove(input_variables['task.label_column'])
ds = dataframe_sampled[columns_copy]
dataframe_sampled_id = compress_and_transfer_dataframe_in_variables(dataframe_sampled)

if model_id is not None and variables.get(model_id) is not None:
    model_compressed = variables.get(model_id)
    model_bin = bz2.decompress(model_compressed)
    with open(MODEL_PATH, "wb") as f:
        f.write(model_bin)
else:
    MODEL_URL = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser("MODEL_URL is None")
    wget.download(MODEL_URL, MODEL_PATH)
print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")

ds.to_csv(SAMPLED_DATA_PATH)
model_file = open(MODEL_PATH, 'rb')
baseline_data = open(SAMPLED_DATA_PATH, 'rb')
files = {'model_file': model_file, 'baseline_data' : baseline_data}
data = {'api_token': SERVICE_TOKEN}

if model_metadata_id is not None and variables.get(model_metadata_id) is not None:
    compressed_model_metadata_json = variables.get(model_metadata_id)
    assert compressed_model_metadata_json is not None
    model_metadata_json = bz2.decompress(compressed_model_metadata_json).decode()
    # dataframe_model_metadata = pd.read_json(model_metadata_json, orient='split')
    dataframe_model_metadata = pd.read_json(model_metadata_json, orient='values')
    print("model_metadata_id: ", model_metadata_id)
    print(dataframe_model_metadata.head())
    data['model_metadata_json'] = model_metadata_json

resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
resultMetadata.put("task.dataframe_sampled_id", dataframe_sampled_id)

try:
    req = requests.post(API_DEPLOY_ENDPOINT, files=files, data=data, verify=False)
    print(req.text)
finally:
    model_file.close()
    baseline_data.close()

print(baseline_data)
print("END " + __file__)
