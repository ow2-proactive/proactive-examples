import bz2
import os
import requests
import sys
import wget
import pandas as pd


def raiser(msg): raise Exception(msg)


MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")

# Get variables
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
    'task.model_metadata_id': None
}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

model_id = input_variables['task.model_id']
model_metadata_id = input_variables['task.model_metadata_id']

if model_id is not None and variables.get(model_id) is not None:
    model_compressed = variables.get(model_id)
    model_bin = bz2.decompress(model_compressed)
    with open(MODEL_PATH, "wb") as f:
        f.write(model_bin)
else:
    MODEL_URL = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser("MODEL_URL is None")
    wget.download(MODEL_URL, MODEL_PATH)
print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")

model_file = open(MODEL_PATH, 'rb')
files = {'model_file': model_file}
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

try:
    req = requests.post(API_DEPLOY_ENDPOINT, files=files, data=data, verify=False)
    print(req.text)
finally:
    model_file.close()
