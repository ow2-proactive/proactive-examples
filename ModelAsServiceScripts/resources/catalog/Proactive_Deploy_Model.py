import sys, bz2, uuid, requests, json, pickle
import pandas as pd
import numpy as np
import time

MODEL_PATH = "last_model.pkl"

#Get Studio variables
if 'variables' in locals():
    if variables.get("MODEL_URL") is not None:
        model_url = variables.get("MODEL_URL")
    if variables.get("API_EXTENSION") is not None:
        api_extension= variables.get("API_EXTENSION")
    if variables.get("LABEL_COLUMN") is not None:
        label= variables.get("LABEL_COLUMN")

api_endpoint = variables.get("DEPLOY_MODEL_ENDPOINT") if variables.get("DEPLOY_MODEL_ENDPOINT") else variables.get("ENDPOINT_MODEL")
api_deploy_endpoint = api_endpoint+api_extension
print("API_DEPLOY_ENDPOINT",api_deploy_endpoint)

service_token = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")

# load the model
input_variables = {'task.model_id': None}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

model_id = input_variables['task.model_id']


if 'variables' in locals():
    if variables.get(model_id) is not None:
        model_compressed = variables.get(model_id)
        model_bin = bz2.decompress(model_compressed)
        with open(MODEL_PATH, "wb") as f:
            model = f.write(model_bin)
        print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")
        MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
    else:
        MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
        wget.download(model_url,MODEL_PATH)

final_model = open(MODEL_PATH, 'rb')
files = {'modelfile':final_model}
data = {'api_token': service_token}
try:
    req = requests.post(api_deploy_endpoint, files=files, data=data)
    print(req.text)
finally:
    final_model.close()