import bz2
import os
import requests
import sys
import wget
import pandas as pd
import urllib.request

def raiser(msg): raise Exception(msg)

#put a randon name
MODEL_PATH = os.path.join(os.getcwd(), "model.zip")

# Get variables
model_name = variables.get("MODEL_NAME") if variables.get("MODEL_NAME") else raiser("MODEL_NAME should be specified")
model_version = variables.get("MODEL_VERSION") if variables.get("APPEND") else None
append = variables.get("APPEND")if variables.get("APPEND") else raiser("APPEND should be specified")
model_url = variables.get("MODEL_URL")if variables.get("MODEL_URL") else None
api_deploy = variables.get("DEPLOY_ENDPOINT") if variables.get("DEPLOY_ENDPOINT") else raiser("DEPLOY_ENDPOINT is None")
service_token = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get(
    "SERVICE_TOKEN_PROPAGATED")
service_endpoint = variables.get("MaaS_DL_INSTANCE_ENDPOINT") if variables.get("MaaS_DL_INSTANCE_ENDPOINT") else variables.get(
    "ENDPOINT_MODEL")
api_deploy_endpoint = service_endpoint + api_deploy
print("[INFO] api_deploy_endpoint: ", api_deploy_endpoint)


if variables.get("zip_file_path") is not None:
    MODEL_PATH = variables.get("zip_file_path")
else:
    model_url = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser("MODEL_URL is None")
    wget.download(model_url, MODEL_PATH)
print('[INFO] model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")

model_file = open(MODEL_PATH, 'rb')
files = {'model_file': model_file }
data = {'api_token': service_token, 'model_name': model_name, 'model_version': model_version, 'append': append}    

try:
    req = requests.post(api_deploy_endpoint, files=files, data=data, verify=False)
    print(req.text)
except Exception as e:
    raiser(e)
finally:
    model_file.close()